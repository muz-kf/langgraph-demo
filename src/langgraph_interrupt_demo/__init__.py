def main() -> None:
    """Run the interactive LangGraph demo."""
    import langgraph_interrupt_demo  # noqa: F401
    # The demo's __main__ block runs on import of the module's __main__
    import runpy
    runpy.run_path("langgraph_interrupt_demo.py", run_name="__main__")
