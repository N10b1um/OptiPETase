load results/5XJH/5xjh_importance.pdb, 5XJH
hide everything, 5XJH
show cartoon, 5XJH
spectrum b, blue_white_red, 5XJH, minimum=-2.0, maximum=2.0
select active_site, 5XJH and resi 131+177+208
show sticks, active_site
color yellow, active_site
select mut_66, 5XJH and resi 66
show sticks, mut_66
color green, mut_66
label (mut_66 and name CA), "K66S"
select mut_233, 5XJH and resi 233
show sticks, mut_233
color green, mut_233
label (mut_233 and name CA), "M233V"
select mut_195, 5XJH and resi 195
show sticks, mut_195
color green, mut_195
label (mut_195 and name CA), "R195G"
select mut_224, 5XJH and resi 224
show sticks, mut_224
color green, mut_224
label (mut_224 and name CA), "K224Y"
select mut_36, 5XJH and resi 36
show sticks, mut_36
color green, mut_36
label (mut_36 and name CA), "A36G"
bg_color white
set ray_shadow, 1
set label_color, black
set label_size, 14
zoom 5XJH
png results/5XJH/plots/5XJH_pareto_selected.png, width=1600, height=1200, dpi=300, ray=1
show surface, 5XJH
set transparency, 0.35
png results/5XJH/plots/5XJH_pareto_selected_surface.png, width=1600, height=1200, dpi=300, ray=1
