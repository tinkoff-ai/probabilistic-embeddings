import argparse
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import wandb


def parse_arguments():
    parser = argparse.ArgumentParser("Download best seed metrics for each group from WandB.")
    parser.add_argument("wandb_path", help="Path to the project in format 'entity/project'.")
    parser.add_argument("-f", "--filename", help="Dump output to file.")
    parser.add_argument("-n", "--num-seeds", help="Number of best seeds to compute statistics for.",
                        type=int, default=5)
    parser.add_argument("--std", help="Show std for each metric.", action="store_true")
    parser.add_argument("--selection-metric", help="Metric to select best seed by.", required=True)
    parser.add_argument("--selection-maximize", help="It true, maximize selection metric. Minimize for false value.",
                        required=True, choices=["true", "false"])
    parser.add_argument("--metric-regexp", nargs="*", help="Regexp to filter metrics.")
    parser.add_argument("--percent", help="Multiply metrics by 100.", action="store_true")
    parser.add_argument("--precision", help="Number of decimal places.", type=int, default=1)
    parser.add_argument("--separator", help="Fields separator.", default=" ")
    parser.add_argument("--url", help="WandB URL.", default="https://api.wandb.ai")
    return parser.parse_args()


def matches(s, regexps):
    for regexp in regexps:
        if re.search(regexp, s) is not None:
            return True
    return False


def get_runs(api, path):
    runs = list(api.runs(path=path))
    runs = [run for run in runs if run.group]
    return runs


def get_metrics(run, metric_regexps=None):
    metrics = run.summary
    if metric_regexps is not None:
        metrics = {k: v for k, v in metrics.items()
                   if matches(k, metric_regexps)}
    return metrics


def prepare_metric(metric, percent=False, precision=2):
    if isinstance(metric, str):
        return metric
    if percent:
        metric = metric * 100
    fmt = "{:." + str(precision) + "f}"
    return fmt.format(metric)


def order_metrics(metrics, metric_regexps=None):
    metrics = list(sorted(list(metrics)))
    if metric_regexps is not None:
        ordered = []
        for regexp in metric_regexps:
            for metric in metrics:
                if metric in ordered:
                    continue
                if matches(metric, [regexp]):
                    ordered.append(metric)
        metrics = ordered
    return metrics


def print_metrics(fp, metrics, run_metrics, separator=" ", percent=False, precision=2, add_std=False):
    print(separator.join(["group"] + list(metrics)), file=fp)
    for run in sorted(list(run_metrics)):
        tokens = [run]
        for name in metrics:
            mean, std = run_metrics[run].get(name, ("N/A", "N/A"))
            mean = prepare_metric(mean, percent=percent, precision=precision)
            std = prepare_metric(std, percent=percent, precision=precision)
            if add_std:
                tokens.append("{} $\pm$ {}".format(mean, std))
            else:
                tokens.append(mean)
        print(separator.join(tokens), file=fp)


def get_best_metrics(runs, num_seeds, metric_regexps, selection_metric, selection_maximize):
    """Returns mean/std metrics for best seeds from each group."""
    if selection_maximize == "true":
        selection_maximize = True
    elif selection_maximize == "false":
        selection_maximize = False
    else:
        raise ValueError(selection_maximize)

    by_group = defaultdict(list)
    for run in runs:
        by_group[run.group].append(run)
    metrics = {}
    for group, runs in by_group.items():
        try:
            runs = list(sorted(runs, key=lambda run: run.summary[selection_metric]))
        except KeyError as e:
            warnings.warn("Group {} doesn't have metric {}.".format(group, selection_metric))
            continue
        if selection_maximize:
            runs = runs[-num_seeds:]
        else:
            runs = runs[:num_seeds]
        by_metric = defaultdict(list)
        for run in runs:
            for k, v in get_metrics(run, metric_regexps).items():
                by_metric[k].append(v)
        metrics[group] = {}
        for name, values in by_metric.items():
            metrics[group][name] = (np.mean(values), np.std(values))
    return metrics


def main(args):
    entity, project = args.wandb_path.split("/")
    api = wandb.apis.public.Api(overrides={"base_url": args.url})

    runs = get_runs(api,"{}/{}".format(entity, project))
    metrics = get_best_metrics(runs, args.num_seeds, args.metric_regexp, args.selection_metric, args.selection_maximize)
    metrics_order = order_metrics(set(sum(map(list, metrics.values()), [])), metric_regexps=args.metric_regexp)
    print_kwargs = {
        "separator": args.separator,
        "percent": args.percent,
        "precision": args.precision,
        "add_std": args.std
    }
    if args.filename is not None:
        with open(args.filename, "w") as fp:
            print_metrics(fp, metrics_order, metrics, **print_kwargs)
    else:
        print_metrics(sys.stdout, metrics_order, metrics, **print_kwargs)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
