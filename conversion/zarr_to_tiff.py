import zarr
import tifffile

def pyramid_to_tiff(zarr_path, out_tiff):
    store = zarr.open(str(zarr_path), mode='r')
    level0 = store["0"]        # 최고 해상도 스케일
    first_key = list(level0.keys())[0] if not isinstance(level0, zarr.core.Array) else None

    arr = (level0[:] if first_key is None else level0[first_key][:])
    tifffile.imwrite(out_tiff, arr)
    print(f"Saved: {out_tiff}")

pyramid_to_tiff("/home/jeongeun.baek/workspace/ws_test/zarr-pipeline/outputs/Well0001/mist_linear_blend/Channel_1_stitched.zarr", "../outputs/Well0001/Well0001_mist_c1_linear_blend.tiff")
