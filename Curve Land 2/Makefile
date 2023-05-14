test:
	cat src.flame | ss=4 qs=10 transparency=0 out=render.png flam3-render

preview: preview_grayscale.png

preview_grayscale.png: preview.png
	convert preview.png -alpha remove -set colorspace Gray -threshold 1 preview_grayscale.png

preview.png: src.flame
	cat src.flame | ss=0.125 qs=1000 transparency=0 out=preview.png flam3-render

release:
	cat src.flame | ss=4 qs=40000 transparency=0 out=render-release.png flam3-render
