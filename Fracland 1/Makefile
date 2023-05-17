preview:
	convert release.png -resize 512x512 preview.png

release: render.png
	../scripts/convert.py render.ong release.png

render.png:
	cat src.flame | ss=8 qs=10000 transparency=1 bits=64 out=render.png flam3-render
