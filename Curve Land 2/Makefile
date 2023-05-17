test:
	cat src.flame | ss=4 qs=10 transparency=0 out=render.png flam3-render

preview: preview-grayscale.png

preview-grayscale.png: preview.png
	convert preview.png -alpha remove -set colorspace Gray -threshold 1 preview-grayscale.png

preview.png: src.flame
	cat src.flame | ss=0.125 qs=1000 transparency=0 out=preview.png flam3-render

release-preview:
	convert release.png -resize 512x512 -threshold 10 release-preview.png

release: release.png

release.png: release-step1.png
	convert release-step1.png -channel A -gamma 1.6 -background black -alpha remove -channel all -fx '(r+g+b)/3' -colorspace Gray release.png

release-step1.png: src.flame
	cat src.flame | ss=4 qs=10000 transparency=1 bits=64 out=release.png flam3-render
