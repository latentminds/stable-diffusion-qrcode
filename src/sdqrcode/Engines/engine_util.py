def init_engine(**engine_kwargs):
    if (
        "hostname" in engine_kwargs
        and engine_kwargs["hostname"] is not None
        and engine_kwargs["hostname"] != ""
    ):
        from . import AutoEngine

        return AutoEngine.AutomaticEngine(**engine_kwargs)
    else:
        from . import DiffusersEngine

        config = engine_kwargs["config"]

        return DiffusersEngine.DiffusersEngine(config=config)
