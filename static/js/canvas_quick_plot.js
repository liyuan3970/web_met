let isDown = false;
let beginPoint = null;
let points = [];
var click_item = 0
const canvas_quick_plot = document.querySelector('#canvas_quick_plot');
const ctx_quick_plot = canvas_quick_plot.getContext('2d');
let canvasW = canvas_quick_plot.width = window.innerWidth * 0.5 * 0.895
let canvasH = canvas_quick_plot.height = window.innerHeight * 0.95 * 0.73
var reback_list = []
let geoCenterX = 0, geoCenterY = 0  // 地图区域的经纬度中心点
let scale = 1   // 地图缩放系数
let geoData = []
let offsetX_qp = 0, offsetY_qp = 0    // 鼠标事件的位置信息
let eventType = ''  // 事件类型
let countour_rect = {
    'x': [],
    'y': []
}
function down(evt) {
    var select_plot_type = $("input[name='taizhou_quick_plot_type']:checked").val()
    if (select_plot_type == 'coutour') {
        isDown = true;
        const { x, y } = getPos(evt);
        points.push({ x, y });
        beginPoint = { x, y };
    }
    else if (select_plot_type == 'line') {
        isDown = true;
        const { x, y } = getPos(evt);
        points.push({ x, y });
        beginPoint = { x, y };

    }
}

function move(evt) {
    var select_plot_type = $("input[name='taizhou_quick_plot_type']:checked").val()
    if (select_plot_type == 'coutour') {
        if (!isDown) return;

        const { x, y } = getPos(evt);
        points.push({ x, y });

        if (points.length > 3) {
            const lastTwoPoints = points.slice(-2);
            const controlPoint = lastTwoPoints[0];
            const endPoint = {
                x: (lastTwoPoints[0].x + lastTwoPoints[1].x) / 2,
                y: (lastTwoPoints[0].y + lastTwoPoints[1].y) / 2,
            }
            drawLine(beginPoint, controlPoint, endPoint);
            beginPoint = endPoint;
        }
    }
    else if (select_plot_type == 'line') {
        if (!isDown) return;

        const { x, y } = getPos(evt);
        points.push({ x, y });

        if (points.length > 3) {
            const lastTwoPoints = points.slice(-2);
            const controlPoint = lastTwoPoints[0];
            const endPoint = {
                x: (lastTwoPoints[0].x + lastTwoPoints[1].x) / 2,
                y: (lastTwoPoints[0].y + lastTwoPoints[1].y) / 2,
            }
            drawLine(beginPoint, controlPoint, endPoint);
            beginPoint = endPoint;
        }

    }
}

function up(evt) {
    var select_plot_type = $("input[name='taizhou_quick_plot_type']:checked").val()
    if (select_plot_type == 'coutour') {
        if (!isDown) return;
        const { x, y } = getPos(evt);
        points.push({ x, y });

        if (points.length > 3) {
            const lastTwoPoints = points.slice(-2);
            const controlPoint = lastTwoPoints[0];
            const endPoint = lastTwoPoints[1];
            drawLine(beginPoint, controlPoint, endPoint);
        }
        beginPoint = null;
        isDown = false;
        points = [];
    }
    else if (select_plot_type == 'line') {
        if (!isDown) return;
        const { x, y } = getPos(evt);
        points.push({ x, y });

        if (points.length > 3) {
            const lastTwoPoints = points.slice(-2);
            const controlPoint = lastTwoPoints[0];
            const endPoint = lastTwoPoints[1];
            drawLine(beginPoint, controlPoint, endPoint);
        }
        beginPoint = null;
        isDown = false;
        points = [];

    }



}

function getPos(evt) {
    return {
        x: evt.offsetX,
        y: evt.offsetY
        // x: evt.clientX,
        // y: evt.clientY
    }
}

function drawLine(beginPoint, controlPoint, endPoint) {
    // ctx_quick_plot.lineWidth = 15;
    var select_plot_type = $("input[name='taizhou_quick_plot_type']:checked").val()
    if (select_plot_type == 'coutour') {
        ctx_quick_plot.lineWidth = 3

    }
    else if (select_plot_type == 'line') {
        ctx_quick_plot.lineWidth = $('#plot_text_size').val()

    }

    ctx_quick_plot.strokeStyle = $("input[name='selected_color']:checked").val()
    countour_rect.x.push(beginPoint.x)
    countour_rect.y.push(beginPoint.y)
    ctx_quick_plot.beginPath();
    ctx_quick_plot.moveTo(beginPoint.x, beginPoint.y);
    ctx_quick_plot.quadraticCurveTo(controlPoint.x, controlPoint.y, endPoint.x, endPoint.y);
    // ctx_quick_plot.lineTo(endPoint.x, endPoint.y);
    ctx_quick_plot.stroke();
    ctx_quick_plot.fill()
    ctx_quick_plot.closePath();
}


// 图像的底图
var maskCanvas = document.createElement('canvas');
// Ensure same dimensions
maskCanvas.width = canvas_quick_plot.width;
maskCanvas.height = canvas_quick_plot.height;
var maskCtx = maskCanvas.getContext('2d');

function drawMap() {
    maskCtx.clearRect(0, 0, canvasW, canvasH)
    // 画布背景
    maskCtx.fillStyle = 'white'
    maskCtx.fillRect(0, 0, canvasW, canvasH)
    drawArea()
    // drawText()
}
// 绘制地图各子区域
function drawArea() {
    let dataArr = geoData.features
    let cursorFlag = false
    for (let i = 0; i < dataArr.length; i++) {
        let centerX = canvasW / 2
        let centerY = canvasH / 2
        dataArr[i].geometry.coordinates.forEach(area => {
            maskCtx.save()
            maskCtx.beginPath()
            maskCtx.strokeStyle = 'black'
            // maskCtx.lineWidth = 3
            maskCtx.translate(centerX, centerY)
            area[0].forEach((elem, index) => {
                let position = toScreenPosition(elem[0], elem[1])
                if (index === 0) {
                    maskCtx.moveTo(position.x, position.y)
                } else {
                    maskCtx.lineTo(position.x, position.y)
                }

            })
            maskCtx.closePath()
            // maskCtx.strokeStyle = 'black'
            maskCtx.lineWidth = 1.5

            maskCtx.fill()
            maskCtx.stroke()
            maskCtx.restore()
        });

    }
}
// 将经纬度坐标转换为屏幕坐标
function toScreenPosition(horizontal, vertical) {
    return {
        x: (horizontal - geoCenterX) * scale,
        y: (geoCenterY - vertical) * scale
    }
}

// 获取包围盒范围，计算包围盒中心经纬度坐标，计算地图缩放系数
function getBoxArea() {
    let N = -90, S = 90, W = 180, E = -180
    geoData.features.forEach(item => {
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
    })
    // 计算包围盒的宽高
    let width = Math.abs(E - W)
    let height = Math.abs(N - S)
    let wScale = canvasW / width
    let hScale = canvasH / height
    // 计算地图缩放系数
    scale = wScale > hScale ? hScale : wScale
    scale = scale - 120
    // 获取包围盒中心经纬度坐标
    geoCenterX = (E + W) / 2
    geoCenterY = (N + S) / 2
}


// Set xor operation
maskCtx.globalCompositeOperation = 'xor';
// Draw the shape you want to take out
geoData = tz_json_county
getBoxArea()
drawMap()
// maskCtx.arc(600, 450, 130, 0, 2 * Math.PI);
maskCtx.fill();
ctx_quick_plot.drawImage(maskCanvas, 0, 0);
reback_list.push(ctx_quick_plot.getImageData(0, 0, canvasW, canvasH))

// 设置线条颜色
ctx_quick_plot.strokeStyle = 'black';
ctx_quick_plot.lineWidth = 3;
ctx_quick_plot.lineJoin = 'round';
ctx_quick_plot.lineCap = 'round';

canvas_quick_plot.addEventListener('mousedown', down, false);
canvas_quick_plot.addEventListener('mousemove', move, false);
canvas_quick_plot.addEventListener('mouseup', up, false);
canvas_quick_plot.addEventListener('mouseout', up, false);


canvas_quick_plot.addEventListener('click',
    function (e) {
        undefined
        offsetX_qp = e.offsetX;
        offsetY_qp = e.offsetY;
        var color_style = $("input[name='selected_color']:checked").val()
        // var canvas_quick_plot_select_value = parseFloat($("input[name='canvas_quick_plot_num']:checked").val());
        console.log(offsetX_qp, offsetY_qp)
        var select_plot_type = $("input[name='taizhou_quick_plot_type']:checked").val()
        if (select_plot_type == 'scatter_icon') {
            console.log("开始绘制图标")
            // select_icon_img[0]
            var plot_icon = new Image()
            plot_icon = select_icon_img[0]
            

            ctx_quick_plot.drawImage(plot_icon, offsetX_qp-25, offsetY_qp-25);

        }
        else if (select_plot_type == 'scatter_pic') {
            console.log("开始绘制色标", 165, 378)//75,398,80,30
        }
        else if (select_plot_type == 'scatter_text') {
            console.log("开始绘制文字")
            
            ctx_quick_plot.font = 'normal 15pt "楷体"';
            ctx_quick_plot.fillText($('#taizhou_quick_plot_text_content').val(), offsetX_qp, offsetY_qp);

        }

    });
// 右键绘图

canvas_quick_plot.oncontextmenu = function (event) {
    var event = event || window.event;
    var select_plot_type = $("input[name='taizhou_quick_plot_type']:checked").val()
    if (select_plot_type == 'coutour') {
        ctx_quick_plot.strokeStyle = $("input[name='selected_color']:checked").val()
        var len_point = countour_rect.x.length
        for (i = 0; i < len_point; i++) {
            if (i == 0) {
                ctx_quick_plot.beginPath();
                ctx_quick_plot.moveTo(countour_rect.x[i], countour_rect.y[i]);
            }
            else if (i != len_point - 1) {
                ctx_quick_plot.lineTo(countour_rect.x[i], countour_rect.y[i]);
            }
            else {
                ctx_quick_plot.lineTo(countour_rect.x[i], countour_rect.y[i]);
                var color_style = $("input[name='selected_color']:checked").val()
                ctx_quick_plot.fillStyle = color_style
                ctx_quick_plot.fill()
                ctx_quick_plot.stroke()
                ctx_quick_plot.closePath();
                countour_rect.x = []
                countour_rect.y = []

            }



        }
        ctx_quick_plot.drawImage(maskCanvas, 0, 0);
        // 设置栈的个数限制为5
        var reback_list_len = reback_list.length
        if (click_item!=0){
            console.log("中间插值")
            reback_list.splice(reback_list_len-click_item,0,ctx_quick_plot.getImageData(0, 0, canvasW, canvasH))
            reback_list.splice(reback_list_len-click_item+1,reback_list_len+click_item-1)

        }
        else{ 
            reback_list.push(ctx_quick_plot.getImageData(0, 0, canvasW, canvasH))
        }
        

        
        // click_item = click_item + 1 
        return false;

    }
    else if (select_plot_type == 'line') {


        ctx_quick_plot.font = 'normal 15pt "楷体"';
        var posx = parseFloat($('#plot_text_clomn').val())
        var posy = parseFloat($('#plot_text_row').val())
        // console.log('开始绘制曲线',posy,posx)
        ctx_quick_plot.fillText($('#plot_text_content').val(), countour_rect.x[0] - posx, countour_rect.y[0] + posy)
        ctx_quick_plot.fillText($('#plot_text_content').val(), countour_rect.x.slice(-1) - posx, countour_rect.y.slice(-1) + posy)
        // ctx_quick_plot.fillText('666',countour_rect.x[0]-40,countour_rect.y[0]+10)
        countour_rect.x = []
        countour_rect.y = []

        // ctx_quick_plot.drawImage(maskCanvas, 0, 0);
        return false;//屏蔽浏览器自带的右键菜单

    }

};
var reback_list_item = 0

$('#reback_canvas').click(function () {
    var reback_list_len = reback_list.length
    if ((reback_list_len - click_item )>=2){        
            ctx_quick_plot.putImageData(reback_list[reback_list_len - click_item - 2], 0, 0);
            click_item = click_item + 1     
            console.log(click_item,reback_list_len )
    }



})

$('#forward_canvas').click(function () {
    var reback_list_len = reback_list.length    
    if ( click_item>0){
        ctx_quick_plot.putImageData(reback_list[reback_list_len-click_item ], 0, 0);
        click_item = click_item - 1;
        console.log(click_item)
        }

})

