# ================================================================
# GeoMEL Grad-CAM Structure Visualization Script (PyMOL)
# ================================================================
# Usage: In PyMOL command line, type   @protein_pymol.pml
#        Or: File -> Run Script -> select this .pml file
# ================================================================

# Load the Grad-CAM annotated PDB
load protein_gradcam.pdb, SidJ

# Basic display settings
hide everything
show cartoon, SidJ
set cartoon_transparency, 0.1
bg_color white

# ========== Core visualization: color by B-factor ==========
# The B-factor column has been replaced with Grad-CAM scores:
#   0      = no contribution   (white)
#   99.99  = highest importance (red)
spectrum b, white_red, SidJ

# ========== Highlight Top-15 key residues ==========
select hotspot, resi 873+872+871+870+869+868+867+866+865+864+863+862+861+860+859
show sticks, hotspot
set stick_radius, 0.2, hotspot

# Labels: show residue name + number on CA atoms
label hotspot and name CA, "%s%s" % (resn, resi)
set label_size, 14
set label_color, black
set label_font_id, 7

# ========== Surface display (optional, shows pocket shape) ==========
# Uncomment the following lines to show a semi-transparent surface:
# show surface, SidJ
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
# Rank 1: Chain A LEU873 (L) GradCAM=0.0000 Neighbors=2
# Rank 2: Chain A ARG872 (R) GradCAM=0.0000 Neighbors=3
# Rank 3: Chain A LYS871 (K) GradCAM=0.0000 Neighbors=4
# Rank 4: Chain A ASP870 (D) GradCAM=0.0000 Neighbors=4
# Rank 5: Chain A THR869 (T) GradCAM=0.0000 Neighbors=4
# Rank 6: Chain A THR868 (T) GradCAM=0.0000 Neighbors=5
# Rank 7: Chain A ARG867 (R) GradCAM=0.0000 Neighbors=5
# Rank 8: Chain A GLU866 (E) GradCAM=0.0000 Neighbors=5
# Rank 9: Chain A SER865 (S) GradCAM=0.0000 Neighbors=5
# Rank 10: Chain A GLU864 (E) GradCAM=0.0000 Neighbors=5
# Rank 11: Chain A PRO863 (P) GradCAM=0.0000 Neighbors=5
# Rank 12: Chain A LYS862 (K) GradCAM=0.0000 Neighbors=6
# Rank 13: Chain A GLU861 (E) GradCAM=0.0000 Neighbors=6
# Rank 14: Chain A SER860 (S) GradCAM=0.0000 Neighbors=6
# Rank 15: Chain A ASP859 (D) GradCAM=0.0000 Neighbors=7
#
# Coverage: 0/873 residues highlighted (0.0%)
# ================================================================
