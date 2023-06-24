def init_engine(**engine_kwargs):
    if (
        "hostname" in engine_kwargs
        and engine_kwargs["hostname"] is not None
        and engine_kwargs["hostname"] != ""
    ):
        import webuiapi
        from . import AutoEngine

        return AutoEngine.AutomaticEngine(**engine_kwargs)
    else:
        from . import DiffusersEngine

        return DiffusersEngine.DiffusersEngine(**engine_kwargs)
