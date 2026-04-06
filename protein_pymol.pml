# ================================================================
# CLEF-GVP Grad-CAM Structure Visualization Script (PyMOL)
# ================================================================
# Usage: In PyMOL command line, type   @protein_pymol.pml
#        Or: File -> Run Script -> select this .pml file
# ================================================================

# Load the Grad-CAM annotated PDB
load protein_gradcam.pdb, Tse5

# Basic display settings
hide everything
show cartoon, Tse5
set cartoon_transparency, 0.1
bg_color white

# ========== Core visualization: color by B-factor ==========
# The B-factor column has been replaced with Grad-CAM scores:
#   0      = no contribution   (white)
#   99.99  = highest importance (red)
spectrum b, white_red, Tse5

# ========== Highlight Top-15 key residues ==========
select hotspot, resi 1127+1129+1115+324+828+1152+1124+177+312+971+49+323+338+45+222
show sticks, hotspot
set stick_radius, 0.2, hotspot

# Labels: show residue name + number on CA atoms
label hotspot and name CA, "%s%s" % (resn, resi)
set label_size, 14
set label_color, black
set label_font_id, 7

# ========== Surface display (optional, shows pocket shape) ==========
# Uncomment the following lines to show a semi-transparent surface:
# show surface, Tse5
# set transparency, 0.7

# ========== Camera setup ==========
orient
zoom hotspot, 15
set ray_shadow, 0
set antialias, 2
set ray_trace_mode, 1

# ========== Export high-quality image (optional) ==========
# ray 2400, 2400
# png gradcam_structure.png, dpi=300

# ================================================================
# Top key residues identified by Grad-CAM:
# Rank 1: Chain A TYR1127 (Y) GradCAM=1.0000 Neighbors=12
# Rank 2: Chain A TYR1129 (Y) GradCAM=0.9956 Neighbors=12
# Rank 3: Chain A TYR1115 (Y) GradCAM=0.9792 Neighbors=10
# Rank 4: Chain A VAL324 (V) GradCAM=0.9789 Neighbors=12
# Rank 5: Chain A VAL828 (V) GradCAM=0.9649 Neighbors=11
# Rank 6: Chain A TYR1152 (Y) GradCAM=0.9636 Neighbors=11
# Rank 7: Chain A TYR1124 (Y) GradCAM=0.9606 Neighbors=12
# Rank 8: Chain A TYR177 (Y) GradCAM=0.9487 Neighbors=11
# Rank 9: Chain A TYR312 (Y) GradCAM=0.9460 Neighbors=12
# Rank 10: Chain A TYR971 (Y) GradCAM=0.9429 Neighbors=12
# Rank 11: Chain A VAL49 (V) GradCAM=0.9389 Neighbors=12
# Rank 12: Chain A VAL323 (V) GradCAM=0.9382 Neighbors=14
# Rank 13: Chain A TYR338 (Y) GradCAM=0.9372 Neighbors=9
# Rank 14: Chain A VAL45 (V) GradCAM=0.9361 Neighbors=8
# Rank 15: Chain A TYR222 (Y) GradCAM=0.9353 Neighbors=13
#
# Coverage: 126/1317 residues highlighted (9.6%)
# ================================================================
