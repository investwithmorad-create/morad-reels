import math, os, io, urllib.request, numpy as np
from PIL import Image
CACHE=os.path.join(os.path.dirname(__file__),'cache')
URL="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
def lonlat_to_px(lon,lat,z):
    n=256*2**z
    x=(lon+180)/360*n
    y=(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n
    return x,y
def tile(z,x,y):
    p=os.path.join(CACHE,f"{z}_{x}_{y}.jpg")
    if not os.path.exists(p):
        req=urllib.request.Request(URL.format(z=z,x=x,y=y),headers={"User-Agent":"morad-map3d/1.0"})
        data=urllib.request.urlopen(req,timeout=30).read()
        open(p,'wb').write(data)
    return Image.open(p).convert('RGB')
def mosaic(lon0,lat0,lon1,lat1,z):
    x0,y0=lonlat_to_px(min(lon0,lon1),max(lat0,lat1),z); x1,y1=lonlat_to_px(max(lon0,lon1),min(lat0,lat1),z)
    tx0,ty0,tx1,ty1=int(x0//256),int(y0//256),int(x1//256),int(y1//256)
    W=(tx1-tx0+1)*256; H=(ty1-ty0+1)*256
    img=Image.new('RGB',(W,H))
    from concurrent.futures import ThreadPoolExecutor
    jobs=[(tx,ty) for ty in range(ty0,ty1+1) for tx in range(tx0,tx1+1)]
    with ThreadPoolExecutor(8) as ex:
        res=list(ex.map(lambda a: tile(z,a[0],a[1]),jobs))
    for (tx,ty),im in zip(jobs,res): img.paste(im,((tx-tx0)*256,(ty-ty0)*256))
    return np.asarray(img), tx0*256, ty0*256
