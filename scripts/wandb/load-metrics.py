import argparse
import re
import sys

import wandb


def parse_arguments():
    parser = argparse.ArgumentParser("Download metrics from WandB.")
    parser.add_argument("wandb_path", help="Path to the project in format 'entity/project'.")
    parser.add_argument("-f", "--filename", help="Dump output to file.")
    parser.add_argument("--group", help="Group to load metrics from (use '-' to match ungrouped runs).")
    parser.add_argument("--run-regexp", nargs="*", help="Regexp to filter runs.")
    parser.add_argument("--metric-regexp", nargs="*", help="Regexp to filter metrics.")
    parser.add_argument("--percent", help="Multiply metrics by 100.", action="store_true")
    parser.add_argument("--precision", help="Number of decimal places.", type=int, default=2)
    parser.add_argument("--separator", help="Fields separator.", default=" ")
    parser.add_argument("--url", help="WandB URL.", default="https://api.wandb.ai")
    return parser.parse_args()


def matches(s, regexps):
    for regexp in regexps:
        if re.search(regexp, s) is not None:
            return True
    return False


def get_runs(api, path, group=None, run_regexps=None):
    runs = list(api.runs(path=path))
    if group == "-":
        runs = [run for run in runs if not run.group]
    elif group is not None:
        runs = [run for run in runs
                if (run.group is not None) and matches(run.group, [group])]
    if run_regexps is not None:
        runs = [run for run in runs
                if matches(run.name, run_regexps)]
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
        assert len(metrics) == len(ordered)
        metrics = ordered
    return metrics


def print_metrics(fp, metrics, run_metrics, separator=" ", percent=False, precision=2):
    print(separator.join(metrics), file=fp)
    for run in sorted(list(run_metrics)):
        tokens = [run] + [prepare_metric(run_metrics[run].get(name, "N/A"), percent=percent, precision=precision)
                          for name in metrics]
        print(separator.join(tokens), file=fp)


def main(args):
    entity, project = args.wandb_path.split("/")
    api = wandb.apis.public.Api(overrides={"base_url": args.url})
    runs = get_runs(api,"{}/{}".format(entity, project), group=args.group, run_regexps=args.run_regexp)
    metrics = {run.name: get_metrics(run, metric_regexps=args.metric_regexp)
               for run in runs}
    metrics_order = order_metrics(set(sum(map(list, metrics.values()), [])), metric_regexps=args.metric_regexp)
    print_kwargs = {
        "separator": args.separator,
        "percent": args.percent,
        "precision": args.precision
    }
    if args.filename is not None:
        with open(args.filename, "w") as fp:
            print_metrics(fp, metrics_order, metrics, **print_kwargs)
    else:
        print_metrics(sys.stdout, metrics_order, metrics, **print_kwargs)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
