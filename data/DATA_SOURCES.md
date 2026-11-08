# 图像来源与使用说明

图像来自 scikit-image 0.26.0 的公开示例数据。各图使用条件分别列于官方 API，不将软件整体许可证自动等同于每张图像的许可。
官方页面：https://scikit-image.org/docs/stable/api/skimage.data.html

仅用于本地课程实验；公开仓库前应逐图确认再分发条件。输入缓存与 PNG 可由本脚本重建。

## camera

Gray-level "camera" image.

Can be used for segmentation and denoising examples.

Returns
-------
camera : (512, 512) uint8 ndarray
    Camera image.

Notes
-----
No copyright restrictions. CC0 by the photographer (Lav Varshney).

.. versionchanged:: 0.18
    This image was replaced due to copyright restrictions. For more
    information, please see [1]_.

References
----------
.. [1] https://github.com/scikit-image/scikit-image/issues/3927

## coffee

Coffee cup.

This photograph is courtesy of Pikolo Espresso Bar.
It contains several elliptical shapes as well as varying texture (smooth
porcelain to coarse wood grain).

Notes
-----
No copyright restrictions.  CC0 by the photographer (Rachel Michetti).

Returns
-------
coffee : (400, 600, 3) uint8 ndarray
    Coffee image.

## astronaut

Color image of the astronaut Eileen Collins.

Photograph of Eileen Collins, an American astronaut. She was selected
as an astronaut in 1992 and first piloted the space shuttle STS-63 in
1995. She retired in 2006 after spending a total of 38 days, 8 hours
and 10 minutes in outer space.

This image was downloaded from the NASA Great Images database
<https://flic.kr/p/r9qvLn>`__.

No known copyright restrictions, released into the public domain.

Returns
-------
astronaut : (512, 512, 3) uint8 ndarray
    Astronaut image.

## chelsea

Chelsea the cat.

An example with texture, prominent edges in horizontal and diagonal
directions, as well as features of differing scales.

Notes
-----
No copyright restrictions.  CC0 by the photographer (Stefan van der Walt).

Returns
-------
chelsea : (300, 451, 3) uint8 ndarray
    Chelsea image.

## coins

Greek coins from Pompeii.

This image shows several coins outlined against a gray background.
It is especially useful in, e.g. segmentation tests, where
individual objects need to be identified against a background.
The background shares enough grey levels with the coins that a
simple segmentation is not sufficient.

Notes
-----
This image was downloaded from the
`Brooklyn Museum Collection
<https://www.brooklynmuseum.org/opencollection/archives/image/51611>`__.

No known copyright restrictions.

Returns
-------
coins : (303, 384) uint8 ndarray
    Coins image.

## moon

Surface of the moon.

This low-contrast image of the surface of the moon is useful for
illustrating histogram equalization and contrast stretching.

Returns
-------
moon : (512, 512) uint8 ndarray
    Moon image.

## page

Scanned page.

This image of printed text is useful for demonstrations requiring uneven
background illumination.

Returns
-------
page : (191, 384) uint8 ndarray
    Page image.

## text

Gray-level "text" image used for corner detection.

Notes
-----
This image was downloaded from Wikipedia
<https://en.wikipedia.org/wiki/File:Corner.png>`__.

No known copyright restrictions, released into the public domain.

Returns
-------
text : (172, 448) uint8 ndarray
    Text image.
