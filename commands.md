Original behavior (same random-drop style):
python main_inpainting_ksvd.py

Explicit random drop fraction:
python main_inpainting_ksvd.py --mask-mode random --missing-frac 0.30 --mask-seed 42

Solid 20x20 face square:
python main_inpainting_ksvd.py --mask-mode face-square --square-size 20 --square-center 44 84

Thick diagonal scratches:
python main_inpainting_ksvd.py --mask-mode scratches --scratch-count 4 --scratch-thickness 5 --mask-seed 42

Combined square + scratches:
python main_inpainting_ksvd.py --mask-mode face-square+scratches --square-size 20 --square-center 44 84 --scratch-count 4 --scratch-thickness 5 --mask-seed 42

Standalone mask preview/export:
python custom_masks.py --mode face-square+scratches --square-size 20 --square-center 44 84 --scratch-count 4 --scratch-thickness 5 --save-mask outputs/custom_mask.npy --save-preview outputs/custom_mask_preview.png

Run full pipeline with text overlay:
python main_inpainting_ksvd.py --mask-mode text-overlay --overlay-text "SAMPLE TEXT"

Tune overlay style:
python main_inpainting_ksvd.py --mask-mode text-overlay --overlay-text "SAMPLE TEXT" --text-scale 0.24 --text-angle -12 --text-stroke 2

Generate only mask preview (standalone):
python custom_masks.py --mode text-overlay --overlay-text "SAMPLE TEXT" --save-mask text_overlay_mask.npy --save-preview text_overlay_preview.png