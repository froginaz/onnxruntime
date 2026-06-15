# Vendored third-party dependencies

## nlohmann/json

- **File:** `nlohmann/json.hpp` (single-header amalgamation)
- **Version:** 3.11.3
- **Source:** https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
- **License:** MIT (SPDX-License-Identifier: MIT) — see the header banner.

Header-only. Used **only** inside the core `.cpp` files (e.g. parsing the
`options_json` load option). It is added to `myaccel_core` as a `PRIVATE` include
and is never exposed through the public headers (`npu_model.h` / `npu_api.h`),
so it does not leak into the adapters or the DLL export surface.

To update: replace `nlohmann/json.hpp` with a newer single-header release and
bump the version above.
