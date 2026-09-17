//! Pillow's bicubic resampler, against Pillow.
//!
//! Reference: tools/dump_resample.py
//!
//! Byte for byte, not "close": the whole point of porting
//! `src/libImaging/Resample.c` rather than writing *a* bicubic filter is that
//! `preprocess_images` feeds the result straight into the tower, so one pixel
//! off by one is a different image than MTPLX saw.

use ornith_rs::refs::RefCase;
use ornith_rs::vision::preprocess::Image;
use ornith_rs::vision::resample::resize_bicubic;

fn cases(case: &RefCase) -> Vec<(usize, usize, usize, usize)> {
    case.meta()
        .unwrap()
        .get("cases")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|c| {
            let v = c.as_arr().unwrap();
            (
                v[0].as_usize().unwrap(),
                v[1].as_usize().unwrap(),
                v[2].as_usize().unwrap(),
                v[3].as_usize().unwrap(),
            )
        })
        .collect()
}

#[test]
fn bicubic_matches_pillow() {
    let Some(case) = RefCase::open_or_skip("resample") else { return };
    for (iw, ih, ow, oh) in cases(&case) {
        let tag = format!("{iw}x{ih}_to_{ow}x{oh}");
        let src = Image {
            width: iw,
            height: ih,
            rgb: case.get(&format!("{tag}_in")).bytes,
        };
        let want = case.get(&format!("{tag}_out")).bytes;
        let got = resize_bicubic(&src, ow, oh);
        assert_eq!((got.width, got.height), (ow, oh), "{tag}: size");
        assert_eq!(got.rgb.len(), want.len(), "{tag}: byte count");
        if got.rgb != want {
            let (i, (g, w)) = got
                .rgb
                .iter()
                .zip(want.iter())
                .enumerate()
                .find(|(_, (g, w))| g != w)
                .unwrap();
            let n = got.rgb.iter().zip(want.iter()).filter(|(g, w)| g != w).count();
            let px = i / 3;
            panic!(
                "{tag}: {n} of {} bytes differ; first at pixel ({}, {}) channel {}: \
                 got {g} want {w}",
                want.len(),
                px % ow,
                px / ow,
                i % 3
            );
        }
    }
}

/// The version this reference was taken from. Nothing else in this port pins
/// a library's internals, so a Pillow upgrade that changed the resampler would
/// otherwise show up as a mysterious drift in the generated description.
#[test]
fn records_the_pillow_version() {
    let Some(case) = RefCase::open_or_skip("resample") else { return };
    let v = case.meta().unwrap().get("pillow_version").unwrap().as_str().unwrap();
    assert!(!v.is_empty());
    eprintln!("reference generated with Pillow {v}");
}
