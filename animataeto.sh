convert -delay 10 -loop 0 $(ls -1 gif/*.png | sort -V) stream.gif
gthumb stream.gif
