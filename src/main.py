from run.runexp import runexp
from args.ehrbase_args import parse_args

if __name__ == "__main__":
    args = parse_args()
    runexp(args)