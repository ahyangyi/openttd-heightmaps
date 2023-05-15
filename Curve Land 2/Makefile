test:
	cat src.flame | ss=4 qs=10 transparency=0 out=render.png flam3-render

preview: preview-grayscale.png

preview-grayscale.png: preview.png
	convert preview.png -alpha remove -set colorspace Gray -threshold 1 preview-grayscale.png

preview.png: src.flame
	cat src.flame | ss=0.125 qs=1000 transparency=0 out=preview.png flam3-render

release: render-release-threshold.png

render-release-threshold.png: render-release-grayscale.png
	convert render-release-grayscale.png -threshold 1 render-release-threshold.png

render-release-grayscale.png: render-release.png
	convert render-release.png -alpha remove -set colorspace Gray render-release-grayscale.png

render-release.png: src.flame
	cat src.flame | ss=4 qs=40000 transparency=0 out=render-release.png flam3-render
