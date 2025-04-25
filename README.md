# CMU DLsys Course Notes and HWs

## Evironment Setup

**Python**:

```bash
# install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh

# set up the evironment
uv sync
```

**If you are using clangd or C/C++ extension in vscode**: Pay attention that the path in the [.clangd](./.clangd) and [c_cpp_properties.json](./.vscode/c_cpp_properties.json) file should be adapted to your own path instead.