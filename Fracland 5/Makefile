preview: release.png
	convert release.png -resize 1024x256 preview.png

release.png: render.png
	../scripts/convert.py render.png release.png

render.png:
	cat src.flame | ss=16 qs=100000 transparency=1 bits=64 out=render.png flam3-render

release-2k.png: render-2k.png
	../scripts/convert.py render-2k.png --delta 0.25 release-2k.png

render-2k.png:
	cat src.flame | ss=2 qs=10000000 transparency=1 bits=64 out=render-2k.png flam3-render
