const self_plot_object = {
    data_canvas: {
        "station_list": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21','22','23','24'],
        "station": [
            [120.5, 28.5, 99],
            [120.92, 28.78, 58.3],
            [120.47, 28.55, 45],
            [120.91, 29.11, 35],
            [121.46, 28.94, 25],
            [121.05, 28.77, 15],
            [121.41, 28.64, 25],
            [120.21, 28.66, 35],
            [121.41, 28.47, 45],
            [121.40, 28.34, 55],
            [121.15, 27.94, 65],
            [121.21, 28.41, 75],
            [120.54, 28.70, 85],
            [121.20, 28.68, 15],
            [120.66, 28.89, 25],
            [121.04, 29.00, 23],
            [121.25, 28.90, 23],
            [121.21, 28.20, 65],
            [121.15, 28.25, 68],
            [120.95, 28.60, 68],
            [121.00, 28.60, 99],
            [121.00, 29.35, 99],
            [121.90, 28.70, 69],
            [122.00, 28.50, 69]

        ]
    },
    data_flush: {
        "station_list": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
        "station": [
            [120.5, 28.5, 99],
            [120.92, 28.78, 58.3],
            [120.47, 28.55, 45],
            [120.91, 29.11, 35],
            [121.46, 28.94, 25],
            [121.05, 28.77, 15],
            [121.41, 28.64, 25],
            [120.21, 28.66, 35],
            [121.41, 28.47, 45],
            [121.40, 28.34, 55],
            [121.15, 27.94, 65],
            [121.21, 28.41, 75],
            [120.54, 28.70, 85],
            [121.20, 28.68, 15],
            [120.66, 28.89, 25],
            [121.04, 29.00, 23],
            [121.25, 28.90, 23],
            [121.21, 28.20, 65],
            [121.15, 28.25, 68],
            [120.95, 28.60, 68],
            [121.00, 28.60, 99]

        ]
    },
    title:function(){
        var titlt_str = $("#self_title_content").val()
        return titlt_str   
    },
    color_bar:function(){
        var father = $('#self_color_select_list')
        // console.log("开始表演",father.children().length)
        if (father.children().length>0){
            var levelV = []
            var labellist = []
            var colorlist = []
            $('.self_color_bar').each(function(){     
                var orgstr =  this.value
                colorlist.push({"fill":this.style.backgroundColor})
                
                if (orgstr){
                    var array = orgstr.split('-')  
                    levelV.push(parseFloat(array[0]))
                    labellist.push(array[1].toString())
                }           
            })
            // 完全正确
            if (colorlist.length == labellist.length) {
                var isobands_options = {
                    zProperty: "value",
                    commonProperties: {
                        "fill-opacity": 0.3
                    },
                    breaksProperties: colorlist
                };
                return {
                    isobands_options:isobands_options,
                    levelV:levelV,
                    labellist:labellist
                }

            }
            // 没有写文字---自定义判断数据大小并分层然后输出
            else {
                var isobands_options = {
                    zProperty: "value",
                    commonProperties: {
                        "fill-opacity": 0.3
                    },
                    breaksProperties: [
                        { fill: "rgb(255,255,255)" },
                        { fill: "rgb(140,246,130)" },
                        { fill: "rgb(0,191,27)" },
                        { fill: "rgb(62,185,255)" },
                        { fill: "rgb(25,0,235)" },
                        { fill: "rgb(255,0,255)" },
                        { fill: "rgb(140,0,65)" }
                    ]
                };
                var labellist = ['0~0.1', '0.1~1', '1~10', '10~25', '25~50', '50~100', '100']
                var levelV = [10, 20, 30, 50, 70, 90, 100, 130];
                return {
                    isobands_options: isobands_options,
                    levelV: levelV,
                    labellist: labellist
                }

            }
        }
        // 完全错误
        else {
            var isobands_options = {
                zProperty: "value",
                commonProperties: {
                    "fill-opacity": 0.3
                },
                breaksProperties: [
                    { fill: "rgb(255,255,255)" },
                    { fill: "rgb(140,246,130)" },
                    { fill: "rgb(0,191,27)" },
                    { fill: "rgb(62,185,255)" },
                    { fill: "rgb(25,0,235)" },
                    { fill: "rgb(255,0,255)" },
                    { fill: "rgb(140,0,65)" }
                ]
            };
            var labellist = ['0~0.1', '0.1~1', '1~10', '10~25', '25~50', '50~100', '100']
            var levelV = [10, 20, 30, 50, 70, 90, 100, 130];
            return {
                isobands_options:isobands_options,
                levelV:levelV,
                labellist:labellist
            }
        }
        


        

        // return "色标"      
    },
    ctx: undefined,
    geoData_self_plot: county_json,
    canvasW_self_plot: undefined,
    canvasH_self_plot: undefined,
    geoCenterX_self_plot: 0,
    geoCenterY_self_plot: 0,
    scale_self_plot_x: 1.0,
    scale_self_plot_y: 1.0,
    map_box: [],
    r: 100,
    download: function (csrf, self_plot_start_time, self_plot_end_time) {
        // 下载数据       
        $.ajax({
            url: "self_plot_download",  // 请求的地址
            type: "post",  // 请求方式
            timeout: 250, //设置延迟上限
            data: {
                'csrfmiddlewaretoken': csrf,
                'self_plot_start_time': self_plot_start_time,
                'self_plot_end_time': self_plot_end_time
            },
            dataType: "json",
            success: function (data) {
                // console.log("下载数据陈工")
                self_plot_object.data_canvas = data.data_canvas
                self_plot_object.data_flush = JSON.parse(JSON.stringify(data.data_canvas))
                
                // data_canvas = data.data_canvas

                // plot_self_flash_data = data.data_canvas
            }
        })
    },
    plot: function (csrf,map,isobandsLay) {
        map.eachLayer(function (layer) {
            if (!layer._container || ('' + jQuery(layer._container).attr('class')).replace(/\s/g, '') != 'leaflet-layer') {
                layer.remove();
            }
        });
        var grid = undefined
        var points = undefined
        var data_canvas = this.data_canvas
        var fea = data_canvas.station.map(i => {
            return {
                type: "Feature",
                properties: {
                    value: i[2].toString()
                },
                geometry: {
                    type: "Point",
                    coordinates: [i[0], i[1]]
                }
            }
        }
        )
        points = turf.featureCollection(fea);
        var interpolate_options = {
            gridType: "points",
            property: "value",
            units: "degrees",
            weight: 10
          };
        grid = turf.interpolate(points, 0.05, interpolate_options);
        
        // var isobands_options = {
        //     zProperty: "value",
        //     commonProperties: {
        //         "fill-opacity": 0.3
        //     },
        //     breaksProperties: [
        //         {fill: "rgb(255,255,255)"},
        //         {fill: "rgb(140,246,130)"},
        //         {fill: "rgb(0,191,27)"},
        //         {fill: "rgb(62,185,255)"},
        //         {fill: "rgb(25,0,235)"},
        //         {fill: "rgb(255,0,255)"},
        //         {fill: "rgb(140,0,65)"}
        //     ]
        // };
        // let levelV = [10, 20, 30, 50, 70, 90, 100,130];
        var isobands_options = self_plot_object.color_bar().isobands_options
        var levelV = self_plot_object.color_bar().levelV


        let isobands = turf.isobands(
            grid,
            levelV,
            isobands_options
        );
        isobandsLay = L.geoJSON(isobands, {
            style: function (feature) {
                return {
                    color: '#4264fb',
                    fillColor: feature.properties.fill,
                    weight: 0.1,
                    fillOpacity: 0.4
                };
            }
        });
        //   裁剪数据
        let features = [];//裁剪后的结果集
        isobands.features.forEach(function (feature1) {
            boundaries.features.forEach(function (feature2) {
                let intersection = null;
                try {
                    intersection = turf.intersect(feature1, feature2);
                } catch (e) {
                    try {
                        //色斑图绘制之后，可能会生成一些非法 Polygon ，例如 在 hole 里存在一些形状（听不懂？去查一下 GeoJSON 的规范），
                        //我遇到的一个意外情况大概是这样，这种 Polygon 在做 intersect() 操作的时候会报错，所以在代码中做了个容错操作。
                        //解决的方法通常就是做一次 turf.buffer() 操作，这样可以把一些小的碎片 Polygon 清理掉。
                        feature1 = turf.buffer(feature1, 0);
                        intersection = turf.intersect(feature1, feature2);
                    } catch (e) {
                        intersection = feature1;//实在裁剪不了就不裁剪了,根据业务需求自行决定
                    }
                }
                if (intersection != null) {
                    intersection.properties = feature1.properties;
                    intersection.id = (Math.random() * 100000).toFixed(0);
                    features.push(intersection);
                }
            });
        });
        //turf.isobands有点不符合业务预期,只有一个等级时,结果集可能为空,无图形显示,写点程序(找出那一个等级，并添加进结果集)补救下
        if (features.length == 0) {
            let maxAttribute = getMaxAttribute(levelV, grid, isobands_options.breaksProperties);
            let value = maxAttribute[0];
            let fill = maxAttribute[1];
            if (value != '' && fill != '') {
                //获取网格点Box
                let gridBox = turf.bbox(grid);
                //生成网格点范围的面
                let gridBoxPolygon = [[[gridBox[0], gridBox[1]], [gridBox[0], gridBox[3]], [gridBox[2], gridBox[3]], [gridBox[2], gridBox[1]], [gridBox[0], gridBox[1]]]];
                //获取网格范围的面与行政边界的交集 Polygon
                let intersectPolygon = null;
                let gridoxFeature = {
                    "type": "Feature",
                    "properties": {"fill-opacity": 0.8},
                    "geometry": {"type": "Polygon", "coordinates": gridBoxPolygon},
                    "id": 10
                };
                try {
                    intersectPolygon = turf.intersect(gridoxFeature, boundaries.features[0]);
                } catch (e) {
                    try {
                        //色斑图绘制之后，可能会生成一些非法 Polygon ，例如 在 hole 里存在一些形状（听不懂？去查一下 GeoJSON 的规范），
                        //我遇到的一个意外情况大概是这样，这种 Polygon 在做 intersect() 操作的时候会报错，所以在代码中做了个容错操作。
                        //解决的方法通常就是做一次 turf.buffer() 操作，这样可以把一些小的碎片 Polygon 清理掉。
                        gridoxFeature = turf.buffer(gridoxFeature, 0);
                        intersectPolygon = turf.intersect(gridoxFeature, boundaries.features[0]);
                    } catch (e) {
                        intersectPolygon = gridoxFeature;//实在裁剪不了就不裁剪了,根据业务需求自行决定
                    }
                }
                //结果添加到结果数组
                if (intersectPolygon != null) {
                    features.push({
                        "type": "Feature",
                        "properties": {"fill-opacity": 0.8, "fill": fill, "value": value},
                        "geometry": intersectPolygon.geometry,
                        "id": 0
                    });
                }
            }
        }
        let intersection = turf.featureCollection(features);
        
        var intersectionLay = L.geoJSON(intersection, {
            style: function (feature) {
                return {
                    color: feature.properties.fill,
                    fillColor: feature.properties.fill,
                    weight: 0.9,
                    fillOpacity: 0.9
                };
            }
        })
        // intersectionLay.remove()
        intersectionLay.addTo(map)
        var lines = L.geoJSON(taizhoulist,{
            style:{
                "color": "black",
                "fillColor": "white",
                "fillOpacity": 0.1,
                "weight": 1,
                "opacity": 0.7
            }
        }).addTo(map)
        // 色标
        function colorlabel(isobands_options, map) {
            var coloritem = isobands_options.breaksProperties.length
            var colorinitlat = 28.4
            var colorinitlon = 120.5
            var colorwidth = 0.15
            var colorheight = 0.4 / coloritem
            var labellist = self_plot_object.color_bar().labellist
            for (var i = 0; i < isobands_options.breaksProperties.length; i++) {
                var textIcon = L.divIcon({
                    html: labellist[i],
                    className: 'label_plot_text'
                });
                var colorlat = colorinitlat - i * colorheight
                var colorlon = colorinitlon
                var latlngs = [[colorlat, colorlon], [colorlat, colorlon + colorwidth], [colorlat - colorheight, colorlon + colorwidth], [colorlat - colorheight, colorlon]]
                var colorstr = isobands_options.breaksProperties[i].fill
                var polygon = L.polygon(latlngs, {
                    color: "black",
                    fillColor: colorstr,
                    fillOpacity: 0.8,
                    weight: 1
                }).addTo(map)
                L.marker([colorlat - colorheight / 3, colorlon + colorwidth + 0.05], { icon: textIcon }).addTo(map);
            }

        }

        colorlabel(isobands_options, map)
        //
        var productIcon = L.divIcon({
            html: self_plot_object.title(),
            className: 'my-div-icon',
            iconSize: [300, 55],

        });

        L.marker([29.4, 120.9], { icon: productIcon }).addTo(map);
        
        //
        self_plot_object.color_bar()
        // 图片
        $.ajax({
            url: "upload_selfplot_data",  // 请求的地址
            type: "post",  // 请求方式
            data: {
                "plot_self_data": JSON.stringify(grid),
                'csrfmiddlewaretoken': csrf
            },
            dataType: "json",
            success: function (data) {
                console.log("upload_selfplot_data is ok")

            }


        })

    },
    flush: function (csrf) { 
        this.data_canvas =  JSON.parse(JSON.stringify(this.data_flush))
    },
    drawMap_self_plot: function () {
        this.ctx.clearRect(0, 0, this.canvasW_self_plot, this.canvasH_self_plot)
        // 画布背景
        this.ctx.fillStyle = 'white'
        this.ctx.fillRect(0, 0, this.canvasW_self_plot, this.canvasH_self_plot)
        this.drawArea_self_plot()
        for (var i = 0; i < this.data_canvas.station_list.length; i++) {
            this.ctx.fillStyle = "black"
            // toScreenPosition
            var canvas_position = this.return_canvas_position(this.data_canvas['station'][i][0], this.data_canvas['station'][i][1])
            // console.log(geoCenterX_self_plot,geoCenterY_self_plot,canvas.width,canvas.height,scale_self_plot,"nwse",map_box)
            this.ctx.fillText(this.data_canvas['station'][i][2].toFixed(1), canvas_position.x, canvas_position.y);
            // ctx.fillText(data_canvas['station'][i][2].toFixed(1), data_canvas['station'][i][0].toFixed(1), data_canvas['station'][i][1]);

        }
    },
    drawArea_self_plot: function () {
        let dataArr = this.geoData_self_plot.features
        let cursorFlag = false
        for (let i = 0; i < dataArr.length; i++) {
            let centerX = this.canvasW_self_plot / 2
            let centerY = this.canvasH_self_plot / 2
            dataArr[i].geometry.coordinates.forEach(area => {
                this.ctx.save()
                this.ctx.beginPath()
                this.ctx.translate(centerX, centerY)
                area[0].forEach((elem, index) => {
                    let position = this.toScreenPosition_self_plot(elem[0], elem[1])
                    if (index === 0) {
                        this.ctx.moveTo(position.x, position.y)
                    } else {
                        this.ctx.lineTo(position.x, position.y)
                    }
                })
                this.ctx.closePath()
                this.ctx.strokeStyle = 'black'
                this.ctx.lineWidth = 1
                this.ctx.fill()
                this.ctx.stroke()
                this.ctx.restore()
            });
            // 动态设置鼠标样式
            if (cursorFlag) {
                canvas.style.cursor = 'pointer'
            } else {
                canvas.style.cursor = 'default'
            }
        }
    },
    toScreenPosition_self_plot: function (horizontal, vertical) {
        return {
            x: (horizontal - this.geoCenterX_self_plot) * this.scale_self_plot_x,
            y: (this.geoCenterY_self_plot - vertical) * this.scale_self_plot_y
        }
    },
    getBoxArea_self_plot: function () {
        let N = -90, S = 90, W = 180, E = -180
        this.geoData_self_plot.features.forEach(item => {
            // 将MultiPolygon和Polygon格式的地图处理成统一数据格式
            if (item.geometry.type === 'Polygon') {
                item.geometry.coordinates = [item.geometry.coordinates]
            }
            // 取四个方向的极值
            item.geometry.coordinates.forEach(area => {
                let areaN = - 90, areaS = 90, areaW = 180, areaE = -180
                area[0].forEach(elem => {
                    if (elem[0] < W) {
                        W = elem[0]
                    }
                    if (elem[0] > E) {
                        E = elem[0]
                    }
                    if (elem[1] > N) {
                        N = elem[1]
                    }
                    if (elem[1] < S) {
                        S = elem[1]
                    }

                })
            })
            // 此处为变相的修改缩放比例
            W = W - 0.06
            E = E + 0.03
            N = N + 0.03
            S = S - 0.06
        })
        // 计算包围盒的宽高
        let width = Math.abs(E - W)
        let height = Math.abs(N - S)
        this.map_box = [width, height, N, W, S, E]
        let wScale = this.canvasW_self_plot / width
        let hScale = this.canvasH_self_plot / height
        // 计算地图缩放系数
        this.scale_self_plot_x = wScale
        this.scale_self_plot_y = hScale   //地图真实的大小
        // scale_self_plot = wScale > hScale ? hScale : wScale
        // scale_self_plot = scale_self_plot //地图真实的大小
        // 获取包围盒中心经纬度坐标
        this.geoCenterX_self_plot = (E + W) / 2
        this.geoCenterY_self_plot = (N + S) / 2
    },
    return_canvas_position: function (lon, lat) {

        return {
            x: (lon - this.map_box[3]) * this.scale_self_plot_x,
            y: (this.map_box[2] - lat) * this.scale_self_plot_y
        }

    },
    run: function (canvas) {
        this.ctx = canvas.getContext('2d');
        canvas.style.backgroundColor = "white";
        var myCanvas_rect = canvas.getBoundingClientRect();
        var widths = myCanvas_rect.width;
        var heights = myCanvas_rect.height;
        this.ctx.font = "15px Arial";
        let offsetX;//通过事件获取鼠标X
        let offsetY;//通过事件获取鼠标Y
        let mouseInCanvas = false
        // let ctx = this.ctx 
        let ballArr = []
        let colorArr = ['red', "blue", 'green']
        this.canvasW_self_plot = canvas.width = widths;
        this.canvasH_self_plot = canvas.height = heights;

        //鼠标移动
        canvas.addEventListener('mousemove',
            function (e) {
                undefined
                mouseInCanvas = true;
                offsetX = e.offsetX;
                offsetY = e.offsetY;

            });
        // 鼠标点击模块
        canvas.addEventListener('click',
            function (e) {
                undefined
                mouseInCanvas = true;
                offsetX = e.offsetX;
                offsetY = e.offsetY;


                var canvas_select_value = parseFloat($("input[name='canvas_num']:checked").val());
                var canvas_select_method = $("#canvas_var  option:selected").val();
                
                for (var i = 0; i < self_plot_object.data_canvas.station.length; i++) {
                    //便利所有自动站中的数据并进行加减  
                    var canvas_position = self_plot_object.return_canvas_position(self_plot_object.data_canvas['station'][i][0], self_plot_object.data_canvas['station'][i][1])


                    var len_r = (offsetX - canvas_position.x) * (offsetX - canvas_position.x) + (offsetY - canvas_position.y) * (offsetY - canvas_position.y);
                    if (canvas_select_method == '+') {

                        if (len_r < self_plot_object.r * self_plot_object.r) {
                            self_plot_object.data_canvas['station'][i][2] = self_plot_object.data_canvas['station'][i][2] + canvas_select_value

                        }
                    }
                    else {

                        if (len_r < self_plot_object.r * self_plot_object.r) {
                            self_plot_object.data_canvas['station'][i][2] = self_plot_object.data_canvas['station'][i][2] - canvas_select_value
                        }
                    }
                }






            });
        // 鼠标滑轮
        canvas.addEventListener('mousewheel',
            function (e) {
                undefined
                mouseInCanvas = true;

                let z = e.deltaY
                if (self_plot_object.r >= 50) {
                    self_plot_object.r = self_plot_object.r + z * 0.1
                }
                else {
                    self_plot_object.r = 50
                }
            });
        //鼠标出画布
        canvas.addEventListener('mouseout',
            function (e) {
                undefined
                mouseInCanvas = false;
            });
        //鼠标进画布
        canvas.addEventListener('mouseover',
            function (e) {
                undefined
                mouseInCanvas = true;
            });


        setInterval(function () {
            undefined
            if (mouseInCanvas) {
                undefined
                //圆圈移动
                // console.log("ball的长度",parseInt(ballArr.length/3))
                if (ballArr.length > 6) {
                    var num_ball = parseInt(ballArr.length / 3)
                    // console.log("ball的长度",num_ball)
                    ballArr.splice(0, num_ball)
                    ballArr.push(new MoveBall(offsetX, offsetY, colorArr[1]));

                }
                else {
                    ballArr.push(new MoveBall(offsetX, offsetY, colorArr[1]));
                }



            }
            self_plot_object.ctx.clearRect(0, 0, canvas.width, canvas.height);
            self_plot_object.getBoxArea_self_plot()
            self_plot_object.drawMap_self_plot()

            // console.log("我在刷新")
            for (let i = 0; i < ballArr.length; i++) {
                undefined
                ballArr[i].render(self_plot_object.ctx);
                ballArr[i].Update(self_plot_object.ctx);
            }
            // console.log(self_plot_object.map_box)
        }, 1);

    }




}


//小球类
class Ball {
    undefined
    constructor(x, y, color) {
        undefined
        this.x = x;
        this.y = y;
        this.color = color;
        this.r = self_plot_object.r
    }
    //绘制小球
    render(ctx) {
        undefined
        ctx.save();
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);

        ctx.lineWidth = 3;
        ctx.strokeStyle = "red";
        ctx.stroke();
        ctx.restore();
    }
}


//会移动的小球类
class MoveBall extends Ball {
    undefined
    constructor(x, y, color) {
        undefined
        super(x, y, color);
    }

    Update(ctx) {
        undefined
        this.x += this.dX;
        this.y += this.dY;
        this.r -= this.dR;
        if (this.r < 0) {
            undefined
            this.r = 0;
        }
    }
}

const canvas = document.getElementById("plot_canvas");


self_plot_object.run(canvas)



