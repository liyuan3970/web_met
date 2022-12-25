const self_plot_object = {
    data_canvas: {
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
    ctx: undefined,
    geoData_self_plot: tz_json,
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
    plot: function (csrf) {
        var data_canvas = this.data_canvas


        // 图片
        $.ajax({
            url: "upload_selfplot_data",  // 请求的地址
            type: "post",  // 请求方式
            data: {
                "plot_self_data": JSON.stringify(data_canvas),
                'csrfmiddlewaretoken': csrf
            },
            dataType: "json",
            success: function (data) {
                // 得到请求的数据
                var plotImg = new Image();
                plotImg.src = data.img
                plotImg.style = "width:100%;height:100%"
                // js构建前端模板
                // var self_plot_div = document.getElementById('self_plot_div')
                $('#self_plot_div').html("")
                $('#self_plot_div').append(plotImg)
                // self_plot_div.append(plotImg)

                // 柱状图
                var chartDom = document.getElementById('self_bar');
                var myChart_bar = echarts.init(chartDom);
                var option;

                option = {
                    tooltip: {
                        trigger: 'axis',
                        axisPointer: {
                            type: 'cross'
                        }
                    },
                    toolbox: {
                        feature: {
                            dataView: { show: true, readOnly: false },
                            restore: { show: true },
                            saveAsImage: { show: true }
                        }
                    },
                    grid: {
                        x: 45,
                        y: 25,
                        x2: 15,
                        y2: 35,
                        borderWidth: 1,
                      },
                    xAxis: {
                        type: 'category',
                        data: ['仙居', '玉环', '三门', '临海', '椒江', '临海', '黄岩']
                    },
                    yAxis: {
                        type: 'value'
                    },
                    series: [
                        {
                            data: [900, 200, 150, 80, 70, 110, 130],
                            type: 'bar'
                        }
                    ]
                };
                option && myChart_bar.setOption(option);
                //表格
                var table_div = document.createElement('div');
                table_div.style = "width:100%;height:100%;overflow:auto"
                var table = document.createElement('table');
                table.setAttribute("class", "table table-bordered")
                var caption = document.createElement('caption')
                caption.innerText = "乡镇排行"
                table_div.append(table)
                table.append(caption)
                table.innerHTML = "<thead><tr><th>排名</th><th>名称</th><th>降水</th></tr></thead><tbody>\
                <tr><td>1</td><td>坎门</td><td>30mm</td></tr>\
                <tr><td>2</td><td>淡竹</td><td>15mm</td></tr>\
                <tr><td>3</td><td>淡竹</td><td>15mm</td></tr>\
                <tr><td>3</td><td>淡竹</td><td>15mm</td></tr>\
                <tr><td>3</td><td>淡竹</td><td>15mm</td></tr>\
                <tr><td>3</td><td>淡竹</td><td>15mm</td></tr>\
                </tbody>"
                $('#self_table').html("")
                $('#self_table').append(table_div)

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
                var canvas_select_method = $("input[name='canvas_var']:checked").val();
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



