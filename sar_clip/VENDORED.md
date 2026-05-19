# Vendored SARCLIP

This package was vendored from the local SARCLIP source tree previously used by
`sfod.cga`:

`/home/storageSDA1/liaojr/SARCLIP/sar_clip`

IRAOD-New imports this local package directly, so CGA no longer depends on
adding `/home/storageSDA1/liaojr/SARCLIP` to `sys.path`.

The SARCLIP model weights are intentionally not vendored. Runtime weight paths
are still controlled by `SARCLIP_PRETRAINED` and `SARCLIP_CACHE_DIR`.
