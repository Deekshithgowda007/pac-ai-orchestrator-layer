def get_instance_ids(inst):
    return (
        inst["0020000E"]["Value"][0],
        inst["00080018"]["Value"][0]
    )
