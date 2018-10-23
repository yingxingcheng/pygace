set title "Fitted Energies"
unset border
unset xtics
unset ytics

set view equal xy 
#center=(.5,sqrt(3)/6,0.0)
#unset key
 splot  './gs_lines.out' u (0.5*(2*$2+$3)/($1+$2+$3)):(sqrt(3)/2*($3)/($1+$2+$3)):($6) t "known gs" w lp pt 4, \
	'./fit.out' u (0.5*(2*$2+$3)/($1+$2+$3)):(sqrt(3)/2*($3)/($1+$2+$3)):($6) t "known str" w p pt 1, \
	'./predstr.out' u (0.5*(2*$2+$3)/($1+$2+$3)):(sqrt(3)/2*($3)/($1+$2+$3)):($5) t "predicted" w p pt 0
pause -1
set title "Calculated Energies"
 splot  './gs_lines.out' u (0.5*(2*$2+$3)/($1+$2+$3)):(sqrt(3)/2*($3)/($1+$2+$3)):($5) t "known gs" w lp pt 4, \
	'./fit.out' u (0.5*(2*$2+$3)/($1+$2+$3)):(sqrt(3)/2*($3)/($1+$2+$3)):($5) t "known str" w p pt 1
pause -1
set title "Calculated and Fitted Energies"
 splot  './fit.out' u (0.5*(2*$2+$3)/($1+$2+$3)):(sqrt(3)/2*($3)/($1+$2+$3)):($5) t "calculated" w p pt 1, \
	'./fit.out' u (0.5*(2*$2+$3)/($1+$2+$3)):(sqrt(3)/2*($3)/($1+$2+$3)):($6) t "fitted" w p pt 3 
pause -1
set border
set xlabel "diameter"
set xtics
set ytics
set ylabel "energy"
set title "ECI vs cluster diameter"
set nokey
set xzeroaxis
set xtics ("pair" 0,"5" 5,"10" 10,"15" 15, "trip" 20,"5" 25,"10" 30,"15" 35, "quad" 40,"5" 45,"10" 50,"15" 55)
plot [-5:60] 'clusinfo.out' u (($1-2)*20.+$2):($4)
pause -1

set title "Residuals of the fit (same order as in fit.out)"
set xtics autofreq
set ylabel "energy"
set xlabel "line number in fit.out"
plot 'fit.out' u ($7)
pause -1
set title "Fitted vs Calculated Energies"
set ylabel "predicted energy"
set xlabel "actual energy"
set nokey
plot \
'fit.out' u ($5):($6) w p pt 1,x w l lt 0
pause -1
