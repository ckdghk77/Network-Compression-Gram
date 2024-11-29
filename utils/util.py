
def get_run_name(args):
    name = ""
    name += "P({}_{})".format(args.prune_amount, args.prune_method)
    name += "_KD({})".format(args.kd_method)
    name += "_SEED({})".format(args.seed)

    return name;

