


let data_canvas = {
    "station_list": ['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'],
    "station": [
        [120.5, 28.5, 99],
        [120.92, 28.78, 55],
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
}


//设置画布
const canvas = document.getElementById("plot_canvas");
const ctx = canvas.getContext('2d');
// canvas.width = 727;
// canvas.height = 651;
let canvasW_self_plot = canvas.width = window.innerWidth *0.5*0.9
let canvasH_self_plot = canvas.height = window.innerHeight *0.95*0.75

// let canvasW_self_plot = canvas.width = 727;
// let canvasH_self_plot = canvas.height = 651;

let geoCenterX_self_plot = 0, geoCenterY_self_plot = 0  // 地图区域的经纬度中心点
let scale_self_plot_x = 1.0   // 地图缩放系数
let scale_self_plot_y = 1.0 
let map_box = []
canvas.style.backgroundColor = "white";

ctx.font = "20px Arial";



let ballArr = [];
let colorArr = ['red', "blue", 'green'];

let mouseInCanvas = false;
let offsetX;//通过事件获取鼠标X
let offsetY;//通过事件获取鼠标Y
let r = 150;//球的半径

let geoData_self_plot = []
geoData_self_plot = tz_json


/////////////////////////////////////////////////////////////////


// 分三步，清空画布、绘制地图各子区域、标注城市名称
function drawMap_self_plot() {
    ctx.clearRect(0, 0, canvasW_self_plot, canvasH_self_plot)
    // 画布背景
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvasW_self_plot, canvasH_self_plot)
    drawArea_self_plot()
    drawText_self_plot()
}

// 绘制地图各子区域
function drawArea_self_plot() {
    let dataArr = geoData_self_plot.features
    let cursorFlag = false
    for (let i = 0; i < dataArr.length; i++) {
        let centerX = canvasW_self_plot / 2
        let centerY = canvasH_self_plot / 2
        dataArr[i].geometry.coordinates.forEach(area => {
            ctx.save()
            ctx.beginPath()
            ctx.translate(centerX, centerY)
            area[0].forEach((elem, index) => {
                let position = toScreenPosition_self_plot(elem[0], elem[1])
                if (index === 0) {
                    ctx.moveTo(position.x, position.y)
                } else {
                    ctx.lineTo(position.x, position.y)
                }
            })
            ctx.closePath()
            ctx.strokeStyle = 'black'
            ctx.lineWidth = 3
            // 将鼠标悬浮的区域设置为橘黄色
            // if (ctx.isPointInPath(offsetX, offsetY)) {
            //     cursorFlag = true
            //     ctx.fillStyle = 'orange'
            //     if (eventType === 'click') {
            //         console.log(dataArr[i])
            //     }
            // } else {
            //     ctx.fillStyle = 'white'
            // }
            ctx.fill()
            ctx.stroke()
            ctx.restore()
        });
        // 动态设置鼠标样式
        if (cursorFlag) {
            canvas.style.cursor = 'pointer'
        } else {
            canvas.style.cursor = 'default'
        }
    }
}
// 标注地图上的城市名称
function drawText_self_plot() {
    let centerX = canvasW_self_plot / 2
    let centerY = canvasH_self_plot / 2
    geoData_self_plot.features.forEach(item => {
        ctx.save()
        ctx.beginPath()
        ctx.translate(centerX, centerY) // 将画笔移至画布的中心
        ctx.fillStyle = '#fff'
        ctx.font = '16px Microsoft YaHei'
        ctx.textAlign = 'center'
        ctx.textBaseLine = 'center'
        let x = 0, y = 0
        //  因不同的geojson文件中中心点属性信息不同，这里需要做兼容性处理
        if (item.properties.cp) {
            x = item.properties.cp[0]
            y = item.properties.cp[1]
        } else if (item.properties.centroid) {
            x = item.properties.centroid[0]
            y = item.properties.centroid[1]
        } else if (item.properties.center) {
            x = item.properties.center[0]
            y = item.properties.center[1]
        }
        let position = toScreenPosition_self_plot(x, y)
        
        ctx.fillText(item.properties.name, position.x, position.y);
        ctx.restore()
    })
}

// 将经纬度坐标转换为屏幕坐标
function toScreenPosition_self_plot(horizontal, vertical) {
    return {
        x: (horizontal - geoCenterX_self_plot) * scale_self_plot_x,
        y: (geoCenterY_self_plot - vertical) * scale_self_plot_y
    }
}

// 获取包围盒范围，计算包围盒中心经纬度坐标，计算地图缩放系数
function getBoxArea_self_plot() {
    let N = -90, S = 90, W = 180, E = -180
    geoData_self_plot.features.forEach(item => {
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
        W = W - 0.03 
        E = E + 0.03
        N = N + 0.03
        S = S - 0.06
    })
    // 计算包围盒的宽高
    let width = Math.abs(E - W)
    let height = Math.abs(N - S)
    map_box = [width,height,N,W,S,E]
    let wScale = canvasW_self_plot / width
    let hScale = canvasH_self_plot / height
    // 计算地图缩放系数
    scale_self_plot_x = wScale  
    scale_self_plot_y = hScale   //地图真实的大小
    // scale_self_plot = wScale > hScale ? hScale : wScale
    // scale_self_plot = scale_self_plot //地图真实的大小
    // 获取包围盒中心经纬度坐标
    geoCenterX_self_plot = (E + W) / 2
    geoCenterY_self_plot = (N + S) / 2
}
//// ////////////////////////////////////////////////////////////////////
// canvas.addEventListener('mousemove', function (event) {
//     offsetX = event.offsetX
//     offsetY = event.offsetY
//     eventType = 'mousemove'
//     ctx.fillStyle = "black"
    
//     // ctx.beginPath();
//     console.log(offsetX, offsetY)
//     // ctx.arc(offsetX, offsetY,150, 0, Math.PI * 2);
// })

// function return_canvas_position(horizontal, vertical) {
//     return {
//         x: (horizontal - geoCenterX_self_plot) * scale_self_plot_x,
//         y: (geoCenterY_self_plot - vertical) * scale_self_plot_y
//     }
// }

function return_canvas_position (lon, lat){

    return {
        x:(lon - map_box[3])*scale_self_plot_x,
        y:(map_box[2]-lat)*scale_self_plot_y
    }

    // return {
    //     x:(lon - map_box[3])*(canvas.width/map_box[0]),
    //     y:(map_box[2]-lat)*(canvas.height/map_box[1])
    // }

}
//小球类
class Ball {
    undefined
    constructor(x, y, color) {
        undefined
        this.x = x;
        this.y = y;
        this.color = color;
        this.r = r;
    }
    //绘制小球
    render() {
        undefined
        ctx.save();
        // getBoxArea()
        // drawMap()
        ctx.beginPath();
        // console.log(this.x, this.y)
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
        
        for (var i = 0; i < data_canvas.station_list.length; i++) {
            ctx.fillStyle="black"
            // toScreenPosition
            var canvas_position = return_canvas_position(data_canvas['station'][i][0], data_canvas['station'][i][1])
            // console.log(geoCenterX_self_plot,geoCenterY_self_plot,canvas.width,canvas.height,scale_self_plot,"nwse",map_box)
            ctx.fillText(data_canvas['station'][i][2].toFixed(1),canvas_position.x, canvas_position.y);
            // ctx.fillText(data_canvas['station'][i][2].toFixed(1), data_canvas['station'][i][0].toFixed(1), data_canvas['station'][i][1]);
            
        }
        ctx.lineWidth = 3;
        ctx.strokeStyle = "red";
        ctx.stroke();
        
        // ctx.fillStyle = this.color;
        // ctx.fill();
        ctx.restore();
    }
}
// {x: 61979.81047555239, y: -56623.36538184083}
// {x: 171.51035538737221, y: -62.93980018038391}

//会移动的小球类
class MoveBall extends Ball {
    undefined
    constructor(x, y, color) {
        undefined
        super(x, y, color);
    }

    Update() {
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
        // console.log(offsetX,offsetY)
        
        var canvas_select_value = parseFloat($("input[name='canvas_num']:checked").val());
        var canvas_select_method = $("input[name='canvas_var']:checked").val();
        for (var i = 0; i < data_canvas.station.length; i++) {
            //便利所有自动站中的数据并进行加减  
            var canvas_position = return_canvas_position(data_canvas['station'][i][0], data_canvas['station'][i][1])
            // var len_r = (offsetX - data_canvas['station'][i][0]) * (offsetX - data_canvas['station'][i][0]) + (offsetY - data_canvas['station'][i][1]) * (offsetY - data_canvas['station'][i][1]);
            var len_r = (offsetX - canvas_position.x) * (offsetX - canvas_position.x) + (offsetY - canvas_position.y) * (offsetY - canvas_position.y);
            if (canvas_select_method == '+') {

                if (len_r < r * r) {
                    data_canvas['station'][i][2] = data_canvas['station'][i][2] + canvas_select_value
                    // console.log("所在位置的经纬度:",offsetX,offsetY,data_canvas['station'][i][0], data_canvas['station'][i][1])
                }
            }
            else {

                if (len_r < r * r) {
                    data_canvas['station'][i][2] = data_canvas['station'][i][2] - canvas_select_value
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
        if (r >= 50) {
            r = r + z * 0.1
        }
        else {
            r = 50
        }
        // r = r + z*0.1

        // console.log(z, r)

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
        if (ballArr.length>100){
            var num_ball = parseInt(ballArr.length/3)
            // console.log("ball的长度",num_ball)
            ballArr.splice(0,num_ball)
            ballArr.push(new MoveBall(offsetX, offsetY, colorArr[1]));

        }
        else {
            ballArr.push(new MoveBall(offsetX, offsetY, colorArr[1]));
        }
        
        

    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    getBoxArea_self_plot()
    drawMap_self_plot()  
            
    // console.log("我在刷新")
    for (let i = 0; i < ballArr.length; i++) {
        undefined
        ballArr[i].render();
        ballArr[i].Update();
    }
},1);