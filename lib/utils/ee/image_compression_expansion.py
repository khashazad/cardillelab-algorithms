import ee


def iterate_convert_image_collection_to_image(new_item, the_structure):
    new_item = ee.Image(new_item)
    the_structure = ee.Image(the_structure)
    return the_structure.addBands(new_item)


def convert_image_collection_to_multi_band_image(image_collection):
    img_col = ee.ImageCollection(image_collection)
    img_col_list = img_col.toList(img_col.size())

    img_col_list = img_col_list.map(lambda item: ee.Image(item))
    img_end_result = ee.Image(
        img_col_list.iterate(iterate_convert_image_collection_to_image, ee.Image(0))
    )
    band_names = img_end_result.bandNames().remove(ee.String("constant"))

    img_end_result = img_end_result.select(band_names)

    return img_end_result


def convert_multi_band_image_to_image_collection(image, labels=None):
    image = ee.Image(image)
    band_names = image.bandNames()

    def convert_band_to_image(band):
        if labels is not None:
            return ee.Image(
                ee.Image(image)
                .select([band])
                .arrayProject([0])
                .arrayFlatten([labels])
                .rename(labels)
            )
        else:
            return ee.Image(ee.Image(image).select([band]).rename("band")).updateMask(
                ee.Image(ee.Image(image).select([band]).rename("band")).gt(-50)
            )

    band_images = band_names.map(convert_band_to_image)

    col = ee.ImageCollection.fromImages(band_images)
    return col
