preview: release.png
	convert release.png -resize 512x512 -threshold 1% preview.png

release.png:
	../scripts/convert.py render.png release.png

render.png:
	cat src.flame | ss=8 qs=10000 transparency=1 bits=64 out=render.png flam3-render
