preview: release.png
	convert release.png -resize 512x512 -threshold 1% preview.png

release.png:
	../scripts/convert.py --alpha 1 --gamma1 3 --gamma2 0.8 --delta 0.11 render.png release.png

# Note: render.png is obtained by running Apophysis with a bunch of plugins...
