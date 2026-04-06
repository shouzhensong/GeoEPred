# ================================================================
# GeoMEL Grad-CAM Structure Visualization Script (PyMOL)
# ================================================================
# Usage: In PyMOL command line, type   @protein_pymol.pml
#        Or: File -> Run Script -> select this .pml file
# ================================================================

# Load the Grad-CAM annotated PDB
load protein_gradcam.pdb, slrp

# Basic display settings
hide everything
show cartoon, slrp
set cartoon_transparency, 0.1
bg_color white

# ========== Core visualization: color by B-factor ==========
# The B-factor column has been replaced with Grad-CAM scores:
#   0      = no contribution   (white)
#   99.99  = highest importance (red)
spectrum b, white_red, slrp

# ========== Highlight Top-15 key residues ==========
select hotspot, resi 765+764+763+762+761+760+759+758+757+756+755+754+753+752+751
show sticks, hotspot
set stick_radius, 0.2, hotspot

# Labels: show residue name + number on CA atoms
label hotspot and name CA, "%s%s" % (resn, resi)
set label_size, 14
set label_color, black
set label_font_id, 7

# ========== Surface display (optional, shows pocket shape) ==========
# Uncomment the following lines to show a semi-transparent surface:
# show surface, slrp
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
# Rank 1: Chain A ARG765 (R) GradCAM=0.0000 Neighbors=3
# Rank 2: Chain A TRP764 (W) GradCAM=0.0000 Neighbors=7
# Rank 3: Chain A TYR763 (Y) GradCAM=0.0000 Neighbors=8
# Rank 4: Chain A ALA762 (A) GradCAM=0.0000 Neighbors=8
# Rank 5: Chain A SER761 (S) GradCAM=0.0000 Neighbors=7
# Rank 6: Chain A MET760 (M) GradCAM=0.0000 Neighbors=9
# Rank 7: Chain A LEU759 (L) GradCAM=0.0000 Neighbors=8
# Rank 8: Chain A SER758 (S) GradCAM=0.0000 Neighbors=6
# Rank 9: Chain A SER757 (S) GradCAM=0.0000 Neighbors=10
# Rank 10: Chain A VAL756 (V) GradCAM=0.0000 Neighbors=10
# Rank 11: Chain A GLU755 (E) GradCAM=0.0000 Neighbors=7
# Rank 12: Chain A LYS754 (K) GradCAM=0.0000 Neighbors=7
# Rank 13: Chain A LYS753 (K) GradCAM=0.0000 Neighbors=6
# Rank 14: Chain A LEU752 (L) GradCAM=0.0000 Neighbors=9
# Rank 15: Chain A LEU751 (L) GradCAM=0.0000 Neighbors=10
#
# Coverage: 0/765 residues highlighted (0.0%)
# ================================================================
