"""SOGC geometry/inference config for an existing RSAR corruption."""

_base_ = './oriented_rcnn_sogc_orthonet_rsar.py'

corrupt = 'chaff'
target_test_img = (
    '/myfile/dataset/RSAR/'
    'corruptions/${corrupt}/test/images/')

# Reuse the clean test annotations and deterministic pipeline; only the image
# prefix changes, matching the current source-free RSAR protocol.
data = dict(
    val=dict(img_prefix=target_test_img),
    test=dict(img_prefix=target_test_img))
