def init_engine(**engine_kwargs):
    if (
        "hostname" in engine_kwargs
        and engine_kwargs["hostname"] is not None
        and engine_kwargs["hostname"] != ""
    ):
        from . import AutoEngine
        
        if "torch_dtype" in engine_kwargs:
            del engine_kwargs["torch_dtype"]

        return AutoEngine.AutomaticEngine(**engine_kwargs)
    else:
        from . import DiffusersEngine

        config = engine_kwargs["config"]
        torch_dtype = engine_kwargs["torch_dtype"]

        return DiffusersEngine.DiffusersEngine(config=config, torch_dtype=torch_dtype)
