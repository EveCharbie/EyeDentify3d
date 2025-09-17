def check_save_name(save_name: str) -> str:
    """
    Checks that the name of the figure contains the extension and returns the format for savefig.
    """
    if "." not in save_name:
        raise RuntimeError("The save_name must contain the figure format (.png, .vsg, ...). ")

    extension = save_name.split(".")[-1]
    return extension
