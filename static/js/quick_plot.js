const quick_plot_object = {
    isDown :false,
    beginPoint :null,
    points :[],
    click_item :0,
    offsetX_qp : 0, 
    offsetY_qp : 0,    
    eventType : '',  
    countour_rect : {
            'x': [],
            'y': []
        },
    canvasW:undefined,
    canvasH:undefined,
    geoData: county_json,
    scale:1.0,
    geoCenterX : 0, 
    geoCenterY : 0 ,
    maskCtx: undefined,
    ctx_quick_plot:undefined,
    maskCanvas:undefined,
    labelinfo:{
        "color": [],
        "text": []
    },
    textinfo:function (){
        //'normal 15pt "楷体"'taizhou_quick_plot_text_size
        var title ='' +$("#taizhou_quick_plot_text_content").val()
        var textstr = $("#taizhou_quick_plot_text_size").val()
        var font = 'normal'+' '+textstr.toString()+'pt "楷体"'

        return {
            fontSize:font,
            titleText:title,
        }
    },
    click_type:function (){
        var select_type = $("input[name='taizhou_quick_plot_type']:checked").val()
        return select_type
    },
    plot_color:function (){
        return $('#view0_color  option:selected').val()
    },
    line_width:function (){
        return $('#view0_width  option:selected').val()
    },
    plot_type:function (){
        return $('#view0_line  option:selected').val()
    },
    drowTextInfo:function () {
        console.log("绘制色标")
        $(".color_bar").each(function (index, el) {
            quick_plot_object.labelinfo.text[index] = el.value;
            quick_plot_object.labelinfo.color[index] = el.style.backgroundColor;
        });

    },
    drawMap:function () {
        this.maskCtx.clearRect(0, 0, this.canvasW, this.canvasH)
        // 画布背景
        this.maskCtx.fillStyle = 'white'
        this.maskCtx.fillRect(0, 0, this.canvasW, this.canvasH)
        this.drawArea()
    },
    drawArea:function () {
        let dataArr = this.geoData.features
        let cursorFlag = false
        for (let i = 0; i < dataArr.length; i++) {
            let centerX = this.canvasW / 2
            let centerY = this.canvasH / 2
            dataArr[i].geometry.coordinates.forEach(area => {
                this.maskCtx.save()
                // this.maskCtx.lineWidth = 5
                this.maskCtx.beginPath()
                this.maskCtx.strokeStyle = 'black'
                
                this.maskCtx.translate(centerX, centerY)
                area[0].forEach((elem, index) => {
                    let position = this.toScreenPosition(elem[0], elem[1])
                    if (index === 0) {
                        this.maskCtx.moveTo(position.x, position.y)
                    } else {
                        this.maskCtx.lineTo(position.x, position.y)
                    }
    
                })
                this.maskCtx.closePath()
                // this.maskCtx.strokeStyle = 'black'
                this.maskCtx.lineWidth = 1.5
    
                this.maskCtx.fill()
                this.maskCtx.stroke()
                this.maskCtx.restore()
            });
    
        }
    },
    // 将经纬度坐标转换为屏幕坐标
    toScreenPosition:function (horizontal, vertical) {
        return {
            x: (horizontal - this.geoCenterX) * this.scale,
            y: (this.geoCenterY - vertical) * this.scale
        }
    },
    
    // 获取包围盒范围，计算包围盒中心经纬度坐标，计算地图缩放系数
    getBoxArea:function () {
        let N = -90, S = 90, W = 180, E = -180
        this.geoData.features.forEach(item => {
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
        let wScale = this.canvasW / width
        let hScale = this.canvasH / height
        // 计算地图缩放系数
        this.scale = wScale > hScale ? hScale : wScale
        this.scale = this.scale - 60
        // 获取包围盒中心经纬度坐标
        this.geoCenterX = (E + W) / 2
        this.geoCenterY = (N + S) / 2
    },
    //相应事件
    getPos:function (evt) {
        return {
            x: evt.offsetX,
            y: evt.offsetY
        }
    },
    move:function (evt) {
        var select_plot_type =  quick_plot_object.plot_type()
        var select_click_type = quick_plot_object.click_type()
        if (select_click_type=='contour'){
            if (select_plot_type == 'contour') {
                if (!quick_plot_object.isDown) return;
        
                const { x, y } = quick_plot_object.getPos(evt);
                quick_plot_object.points.push({ x, y });
        
                if (quick_plot_object.points.length > 3) {
                    const lastTwoPoints = quick_plot_object.points.slice(-2);
                    const controlPoint = lastTwoPoints[0];
                    const endPoint = {
                        x: (lastTwoPoints[0].x + lastTwoPoints[1].x) / 2,
                        y: (lastTwoPoints[0].y + lastTwoPoints[1].y) / 2,
                    }
                    quick_plot_object.drawLine(quick_plot_object.beginPoint, controlPoint, endPoint);
                    quick_plot_object.beginPoint = endPoint;
                }
            }
            else if (select_plot_type == 'line') {
                if (!quick_plot_object.isDown) return;
        
                const { x, y } = quick_plot_object.getPos(evt);
                quick_plot_object.points.push({ x, y });
        
                if (quick_plot_object.points.length > 3) {
                    const lastTwoPoints = quick_plot_object.points.slice(-2);
                    const controlPoint = lastTwoPoints[0];
                    const endPoint = {
                        x: (lastTwoPoints[0].x + lastTwoPoints[1].x) / 2,
                        y: (lastTwoPoints[0].y + lastTwoPoints[1].y) / 2,
                    }
                    quick_plot_object.drawLine(this.beginPoint, controlPoint, endPoint);
                    quick_plot_object.beginPoint = endPoint;
                }
        
            }

        }
        
    },
    
    up:function (evt) {
        var select_plot_type = quick_plot_object.plot_type()
        var select_click_type = quick_plot_object.click_type()
        if (select_click_type=='contour'){
            if (select_plot_type == 'contour') {
                if (!quick_plot_object.isDown) return;
                const { x, y } = quick_plot_object.getPos(evt);
                quick_plot_object.points.push({ x, y });
        
                if (quick_plot_object.points.length > 3) {
                    const lastTwoPoints = quick_plot_object.points.slice(-2);
                    const controlPoint = lastTwoPoints[0];
                    const endPoint = lastTwoPoints[1];
                    quick_plot_object.drawLine(quick_plot_object.beginPoint, controlPoint, endPoint);
                }
                quick_plot_object.beginPoint = null;
                quick_plot_object.isDown = false;
                quick_plot_object.points = [];
            }
            else if (select_plot_type == 'line') {
                if (!quick_plot_object.isDown) return;
                const { x, y } = quick_plot_object.getPos(evt);
                quick_plot_object.points.push({ x, y });
        
                if (quick_plot_object.points.length > 3) {
                    const lastTwoPoints = quick_plot_object.points.slice(-2);
                    const controlPoint = lastTwoPoints[0];
                    const endPoint = lastTwoPoints[1];
                    quick_plot_object.drawLine(quick_plot_object.beginPoint, controlPoint, endPoint);
                }
                quick_plot_object.beginPoint = null;
                quick_plot_object.isDown = false;
                quick_plot_object.points = [];
        
            }
            
        }
        
    
    
    
    },
    down:function (evt) {
        var select_plot_type = quick_plot_object.plot_type()
        var select_click_type = quick_plot_object.click_type()
        if (select_click_type=='contour'){
            if (select_plot_type == 'contour') {
                quick_plot_object.isDown = true;
                const { x, y } = quick_plot_object.getPos(evt);
                quick_plot_object.points.push({ x, y });
                quick_plot_object.beginPoint = { x, y };
            }
            else if (select_plot_type == 'line') {
                quick_plot_object.isDown = true;
                const { x, y } = quick_plot_object.getPos(evt);
                quick_plot_object.points.push({ x, y });
                quick_plot_object.beginPoint = { x, y };
        
            }
            
        }
        
    },
    
    
    drawLine:function (beginPoint, controlPoint, endPoint) {
        // ctx_quick_plot.lineWidth = 15;
        quick_plot_object.ctx_quick_plot.strokeStyle = quick_plot_object.plot_color()
        var select_plot_type = quick_plot_object.plot_type()
        var select_click_type = quick_plot_object.click_type()
        if (select_click_type=='contour'){
            if (select_plot_type == 'contour') {
                quick_plot_object.ctx_quick_plot.lineWidth = quick_plot_object.line_width()
        
            }
            else if (select_plot_type == 'line') {
                quick_plot_object.ctx_quick_plot.lineWidth = quick_plot_object.line_width()
        
            }
        
            
            quick_plot_object.countour_rect.x.push(quick_plot_object.beginPoint.x)
            quick_plot_object.countour_rect.y.push(quick_plot_object.beginPoint.y)
            quick_plot_object.ctx_quick_plot.beginPath();
            quick_plot_object.ctx_quick_plot.moveTo(quick_plot_object.beginPoint.x, quick_plot_object.beginPoint.y);
            quick_plot_object.ctx_quick_plot.quadraticCurveTo(controlPoint.x, controlPoint.y, endPoint.x, endPoint.y);
            quick_plot_object.ctx_quick_plot.lineTo(endPoint.x, endPoint.y);
            quick_plot_object.ctx_quick_plot.stroke();
            //quick_plot_object.ctx_quick_plot.fill()
            quick_plot_object.ctx_quick_plot.closePath();
            
        }
        
    },
    click_fun: function (e) {
        undefined
        this.offsetX_qp = e.offsetX;
        this.offsetY_qp = e.offsetY;
        // var canvas_quick_plot_select_value = parseFloat($("input[name='canvas_quick_plot_num']:checked").val());
        var select_plot_type = quick_plot_object.click_type()//'none'//$("input[name='taizhou_quick_plot_type']:checked").val()
        if (select_plot_type == 'scatter_icon') {
            // console.log("开始绘制图标")
            var plot_icon = new Image()
            plot_icon = select_icon_img[0]
            quick_plot_object.ctx_quick_plot.drawImage(plot_icon, e.offsetX-35, e.offsetY-plot_icon.height/2);

        }
        else if (select_plot_type == 'scatter_pic') {
            // console.log("开始绘制色标")//75,398,80,30
            quick_plot_object.drowTextInfo()
            quick_plot_object.ctx_quick_plot.beginPath();
            for (i = 0; i < quick_plot_object.labelinfo.color.length; i++) {
                quick_plot_object.ctx_quick_plot.fillStyle = quick_plot_object.labelinfo.color[i]
                quick_plot_object.ctx_quick_plot.fillRect(e.offsetX, e.offsetY + i * 30, 80, 30)
                quick_plot_object.ctx_quick_plot.font = 'normal 15pt "楷体"'
                quick_plot_object.ctx_quick_plot.fillStyle = "black"
                quick_plot_object.ctx_quick_plot.fillText(quick_plot_object.labelinfo.text[i], e.offsetX + 80 + 8,  e.offsetY + i * 30 + 18)       
            }
            quick_plot_object.ctx_quick_plot.closePath();
        }
        else if (select_plot_type == 'scatter_text') {
            // console.log("开始绘制文字")
            quick_plot_object.ctx_quick_plot.font = quick_plot_object.textinfo().fontSize//'normal 15pt "楷体"';//$('#taizhou_quick_plot_text_content').val()
            quick_plot_object.ctx_quick_plot.fillStyle = 'black'
            quick_plot_object.ctx_quick_plot.fillText(quick_plot_object.textinfo().titleText, e.offsetX, e.offsetY);
            quick_plot_object.ctx_quick_plot.fillStyle = quick_plot_object.plot_color()

        }
    },
    right_fun:function (event) {
        var event = event || window.event;
        var select_plot_type = quick_plot_object.plot_type()
        if (select_plot_type == 'contour') {
            quick_plot_object.ctx_quick_plot.strokeStyle = quick_plot_object.plot_color()
            var len_point = quick_plot_object.countour_rect.x.length
            for (i = 0; i < len_point; i++) {
                if (i == 0) {
                    quick_plot_object.ctx_quick_plot.beginPath();
                    quick_plot_object.ctx_quick_plot.moveTo(quick_plot_object.countour_rect.x[i], quick_plot_object.countour_rect.y[i]);
                }
                else if (i != len_point - 1) {
                    quick_plot_object.ctx_quick_plot.lineTo(quick_plot_object.countour_rect.x[i], quick_plot_object.countour_rect.y[i]);
                }
                else {
                    quick_plot_object.ctx_quick_plot.lineTo(quick_plot_object.countour_rect.x[i], quick_plot_object.countour_rect.y[i]);
                    //var color_style = 'red'//$("input[name='selected_color']:checked").val()
                    quick_plot_object.ctx_quick_plot.fillStyle = quick_plot_object.plot_color()
                    quick_plot_object.ctx_quick_plot.lineWidth = quick_plot_object.line_width()
                    quick_plot_object.ctx_quick_plot.fill()
                    quick_plot_object.ctx_quick_plot.stroke()
                    quick_plot_object.ctx_quick_plot.closePath();
                    quick_plot_object.countour_rect.x = []
                    quick_plot_object.countour_rect.y = []
    
                }
    
    
    
            }
            quick_plot_object.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
            quick_plot_object.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
            quick_plot_object.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
            quick_plot_object.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
            quick_plot_object.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
            quick_plot_object.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
            quick_plot_object.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
            // 设置栈的个数限制为5
            // var reback_list_len = reback_list.length
            // if (click_item!=0){
            //     console.log("中间插值")
            //     reback_list.splice(reback_list_len-click_item,0,ctx_quick_plot.getImageData(0, 0, canvasW, canvasH))
            //     reback_list.splice(reback_list_len-click_item+1,reback_list_len+click_item-1)
    
            // }
            // else{ 
            //     reback_list.push(ctx_quick_plot.getImageData(0, 0, canvasW, canvasH))
            // }
            
    
            
            // click_item = click_item + 1 
            return false;
    
        }
        else if (select_plot_type == 'line') {
    
    
            // quick_plot_object.ctx_quick_plot.font = 'normal 15pt "楷体"';
            // var posx = parseFloat($('#plot_text_clomn').val())
            // var posy = parseFloat($('#plot_text_row').val())
            // // console.log('开始绘制曲线',posy,posx)
            // ctx_quick_plot.fillText($('#plot_text_content').val(), countour_rect.x[0] - posx, countour_rect.y[0] + posy)
            // ctx_quick_plot.fillText($('#plot_text_content').val(), countour_rect.x.slice(-1) - posx, countour_rect.y.slice(-1) + posy)
            // ctx_quick_plot.fillText('666',countour_rect.x[0]-40,countour_rect.y[0]+10)
            quick_plot_object.countour_rect.x = []
            quick_plot_object.countour_rect.y = []
    
            // ctx_quick_plot.drawImage(maskCanvas, 0, 0);
            return false;//屏蔽浏览器自带的右键菜单
    
        }
    
    },

    run: function (quick_plot) {
        
        // quick_plot.style.backgroundColor = "black";
        this.ctx_quick_plot = quick_plot.getContext('2d');
        var widths = $('#quick_plot').actual('width')
        var heights = $('#quick_plot').actual('height')
        this.canvasW = quick_plot.width = widths;
        this.canvasH = quick_plot.height = heights;

        this.maskCanvas = document.createElement('canvas');
        this.maskCtx = this.maskCanvas.getContext('2d');
        this.maskCanvas.width = this.canvasW
        this.maskCanvas.height = this.canvasH 

        
        this.ctx_quick_plot.lineWidth = quick_plot_object.line_width()
        this.maskCtx.globalCompositeOperation = 'xor';
        this.getBoxArea()

        
        this.drawMap()
        this.ctx_quick_plot.drawImage(this.maskCanvas, 0, 0);
        this.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
        this.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
        this.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
        this.ctx_quick_plot.drawImage(quick_plot_object.maskCanvas, 0, 0);
        


        //this.ctx_quick_plot.strokeStyle = 'black';
        this.ctx_quick_plot.lineWidth = 3;
        this.ctx_quick_plot.lineJoin = 'round';
        this.ctx_quick_plot.lineCap = 'round';

        quick_plot.addEventListener('mousedown', quick_plot_object.down, false);
        quick_plot.addEventListener('mousemove', quick_plot_object.move, false);
        quick_plot.addEventListener('mouseup', quick_plot_object.up, false);
        quick_plot.addEventListener('mouseout', quick_plot_object.up, false);
        quick_plot.addEventListener('click', quick_plot_object.click_fun, false);
        quick_plot.oncontextmenu = this.right_fun

    }


}
    

const quick_plot = document.getElementById("quick_plot");


quick_plot_object.run(quick_plot)



