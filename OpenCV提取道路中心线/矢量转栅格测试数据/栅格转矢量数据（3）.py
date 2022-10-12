from osgeo import gdal, ogr, osr
import os
import cv2
from skimage import morphology
import numpy as np
import datetime


def shp_to_tiff(shp_file, refore_tif, output_tif):
    """
     将shp文件转换成tiff文件，用来去黑边
    :param shp_file: 矢量数据
    :param refore_tif:参考栅格（用于获取投影坐标信息）
    :param output_tiff: 转换的tiff结
    :param projection:  投影坐标信息
    :param transform:   仿射变换信息
    :param rows:
    :param cols:
    :return: 矢量栅格图像
    """


    # open origin image
    data = gdal.Open(refore_tif, gdal.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    geo_projection = data.GetProjection()

    x_res = data.RasterXSize
    y_res = data.RasterYSize

    # set out iamge
    y_ds = gdal.GetDriverByName('GTiff').Create(output_tif, x_res, y_res, 1, gdal.GDT_Byte,
                                                options=['COMPRESS=LZW', 'BIGTIFF=YES'])
    y_ds.SetGeoTransform(geo_transform)
    y_ds.SetProjection(geo_projection)

    # set out memory
    target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    # open vector
    datasource = ogr.Open(shp_file)
    layer = datasource.GetLayer()
    extent = layer.GetExtent()  # 上下左右边界
    top_left = [extent[0], extent[3]]
    bottom_right = [extent[1], extent[2]]
    if len(layer) > 0:
        gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])
    y_buffer = band.ReadAsArray()
    y_ds.WriteRaster(0, 0, x_res, y_res, y_buffer.tostring())
    target_ds = None
    y_ds = None
    return top_left,bottom_right

def ext_road(output_tif, refore_tif, centerline_tif):
# def ext_road(path, outfilename="r1.tif"):
    '''影像道路提取中心线，同级目录输出文件
    :param path: 输入栅格文件路径
    :param outfilename: 输出栅格文件名
    '''
    # 读取灰度图片，并显示
    img = cv2.imread(output_tif, 0)  # 直接读为灰度图像
    img[img == 255] = 1
    skeleton0 = morphology.skeletonize(img)
    skeleton = skeleton0.astype(np.uint8) * 255
    # # cv2.imshow('image', skeleton)
    # # cv2.waitKey(0)
    # filepath = os.path.dirname(path) + "\\" + outfilename




    cv2.imwrite(centerline_tif, skeleton)


def get_contour(img):
    """获取连通域
    :param img: opencv读取的图片数组
    :return: 连通域(拐点坐标)
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return img_gray, contours


def registration(input_path, out_path, top_left, bottom_right, ik, jk, srs):
    """
    基于python GDAL配准
    :param input_path: 需要配准的栅格文件
    :param out_path: 输出配准后的栅格文件位置
    :param top_left: 左上角坐标
    :param bottom_right: 右下角坐标
    :param ik: 行空白分辨率
    :param jk: 列空白分辨率
    :return:
    """
    # 打开栅格文件
    dataset = gdal.Open(input_path, gdal.GA_Update)
    # 获取图片的实际分辨率
    img = cv2.imread(input_path, 1)
    x, y = img.shape[1], img.shape[0]
    # print('配准的左上角、右下角、x轴白边、y轴白边、像素x、像素y,坐标系', top_left, bottom_right, ik, jk, x, y, srs,
    #       type(srs))
    # 构造控制点列表 gcps_list
    gcps_list = [gdal.GCP(top_left[0], top_left[1], 0, 0, 0),
                 gdal.GCP(bottom_right[0], top_left[1], 0, x - jk, 0),
                 gdal.GCP(top_left[0], bottom_right[1], 0, 0, y - ik),
                 gdal.GCP(bottom_right[0], bottom_right[1], 0, x - jk, y - ik)]
    # 设置空间参考
    spatial_reference = osr.SpatialReference()
    if srs == 4528:
        spatial_reference.SetWellKnownGeogCS('CGCS2000')
    else:
        spatial_reference.ImportFromEPSG(srs)
    # 添加控制点
    dataset.SetGCPs(gcps_list, spatial_reference.ExportToWkt())
    # tps校正 重采样:最邻近法
    dst_ds = gdal.Warp(out_path, dataset, format='GTiff', tps=True, width=x, height=y,
                       resampleAlg=gdal.GRIORA_NearestNeighbour)


def rasterToLine(centerline_tif, shpPath, image_path):
    '''栅格转为矢量线
    :param path: 输入栅格文件路径
    :param shpPath: 输出要素文件路径
    '''
    # 1.导入图片
    img_src = cv2.imread(centerline_tif)

    # 2.获取连通域
    img_gray, contours = get_contour(img_src)
    # 创建线要素
    # driver = ogr.GetDriverByName('ESRI Shapefile')
    # dataSource=driver.CreateDataSource(path)
    # layer =dataSource.CreateLayer("line1")
    # fieldDefn = ogr.FieldDefn('id', ogr.OFTString)
    # fieldDefn.SetWidth(4)
    # layer.CreateField(fieldDefn)

    inimg = gdal.Open(image_path)      #读取原始有地理坐标的文件
    Geoimg = inimg.GetGeoTransform()   #读取仿射矩阵信息

    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 为了使属性表字段支持中文
    strVectorFile = shpPath  # 定义写入路径及文件名
    ogr.RegisterAll()  # 注册所有的驱动
    strDriverName = "ESRI Shapefile"  # 创建数据，这里创建ESRI的shp文件
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print("%s 驱动不可用！\n", strDriverName)

    oDS = oDriver.CreateDataSource(strVectorFile)  # 创建数据源
    if oDS == None:
        print("创建文件【%s】失败！", strVectorFile)

    srs = osr.SpatialReference()  # 创建空间参考
    srs.ImportFromEPSG(4549)  # 定义地nnnnnnnnnnnnnnnnnnn理坐标系WGS1984
    papszLCO = []
    # 创建图层，创建一个线图层,"TestPolygon"->属性表名
    oLayer = oDS.CreateLayer("Testline", srs, ogr.wkbLineString, papszLCO)
    if oLayer == None:
        print("图层创建失败！\n")
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(4326)
    # papszLCO = []
    # oLayer = oDS.CreateLayer("TestPolygon", srs, ogr.wkbPolygon, papszLCO)

    '''下面添加矢量数据，属性表数据、矢量数据坐标'''
    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建一个叫FieldID的整型属性
    oLayer.CreateField(oFieldID, 1)

    oFieldName = ogr.FieldDefn("FieldName", ogr.OFTString)  # 创建一个叫FieldName的字符型属性
    oFieldName.SetWidth(100)  # 定义字符长度为100
    oLayer.CreateField(oFieldName, 1)

    oDefn = oLayer.GetLayerDefn()  # 定义要素
    # lines = ogr.Geometry(ogr.wkbLinearRing)  # 定义总的线集

    # 创建一条线
    oFeature = ogr.Feature(oDefn)
    oFeature.SetField(0, 0)  # 第一个参数表示第几个字段，第二个参数表示字段的值
    oFeature.SetField(1, "line1")

    for contour in contours:
        box1 = ogr.Geometry(ogr.wkbLinearRing)
        for point in contour:
            x_col = Geoimg[0] + Geoimg[1] * (float(point[0, 0])) + (float(point[0, 1])) * Geoimg[2]
            y_row = Geoimg[3] + Geoimg[4] * (float(point[0, 0])) + (float(point[0, 1])) * Geoimg[5]
            box1.AddPoint(x_col, y_row)
        oFeature.SetGeometry(box1)
        oLayer.CreateFeature(oFeature)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    # 创建WKT 文本点
    for i in range(10):
        wkt = "LINESTRING(%f %f,%f %f)" % (
            float(0), float(1000 + i * 10), float(1000), float(1000 + i * 10))
        # 生成实体点
        point = ogr.CreateGeometryFromWkt(wkt)
        oFeature.SetGeometry(point)
        oFeature.SetGeometry(ring)
        oLayer.CreateFeature(oFeature)

    oFeature = None
    oDS = None


if __name__ == "__main__":
    input_shp = '道路.shp'
    refore_tif = 'road.tif'
    output_tif = 'output_tif'
    Centerline_tif = 'centerline.tif'
    Centerline_prj_tif = 'centerline_prj.tif'
    output_shapfile = 'output_Certerline.shp'

    start_time = datetime.datetime.now()

    top_left, bottom_right = shp_to_tiff(input_shp, refore_tif, output_tif)
    ext_road(output_tif, refore_tif, Centerline_tif)
    registration(Centerline_tif, Centerline_prj_tif, top_left, bottom_right, 0, 0, 4529)
    rasterToLine(Centerline_prj_tif, output_shapfile, refore_tif)

    end_time = datetime.datetime.now()
    print("Succeeded at", end_time)
    print("Elapsed Time:", end_time - start_time)  # 输出程序运行所需时间


