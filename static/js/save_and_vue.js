//  这是前端预览文档的核心代码
function head_div(data) {
    console.log(data['type'])
    var div_main = document.createElement('div')
    div_main.style = "height:27%;width:100%;background-color:white;position: relative;"

    var div_type = document.createElement('div')
    div_type.style = "top:30%;height:10%;width:100%;background-color:white;position: absolute;text-align: center;font-size:50px ;font-family:'宋体';color:red"
    div_type.innerText = data['type']

    var div_index = document.createElement('div')
    div_index.style = "top:60%;height:10%;width:100%;background-color:white;position: absolute;text-align: center;font-size:20px ;font-family:'宋体';color:black"
    div_index.innerText = "第" + data['index'] + "期"

    var div_company = document.createElement('div')
    div_company.style = "top:80%;;height:10%;width:100%;position: absolute;text-align: center;font-size:20px ;font-family:'宋体';color:black"
    div_company.innerText = data['company'] 

    var div_time = document.createElement('div')
    div_time.style = "top:80%;left:10% ;height:10%;width:100%;position: absolute;font-size:20px ;font-family:'宋体';color:black"
    div_time.innerText =  "2022年3月01日19时"

    var div_writer = document.createElement('div')
    div_writer.style = "top:65%;left:80%;height:10%;width:100%;position: absolute;font-size:20px ;font-family:'宋体';color:black"
    div_writer.innerText = "撰稿人:" +data['writer'] 

    var div_putter = document.createElement('div')
    div_putter.style = "top:80%;left:80%;height:10%;width:100%;position: absolute;font-size:20px ;font-family:'宋体';color:black"
    div_putter.innerText = "签发人:" +data['putter'] 

    var div_red_line = document.createElement('div')
    div_red_line.style = "top:95%;left:8%;height:1%;width:88%;;position: absolute;border: 2px solid red;"
        
    div_main.append(div_type)
    div_main.append(div_index)
    div_main.append(div_company)
    div_main.append(div_time)
    div_main.append(div_writer)
    div_main.append(div_putter)
    div_main.append(div_red_line)

    $('#product_view').append(div_main)
}
function text_div(data) {
    console.log(data['type'])
    var div_main = document.createElement('div')
    div_main.style = "display:flex; justify-content:center align-items:center;height:auto;width:100%;background-color:white;font-size:20px ;font-family:'宋体';color:black;"

    var div_blank = document.createElement('div')
    div_blank.style = "width: 10%;"

    var div_content_main = document.createElement('div')
    div_content_main.style = "width: 80%;"

    var div_title = document.createElement('div')
    div_title.style = ";width:100% ;line-height:50px;font-size:25px ;font-family:'宋体';color:black;text-align: center;font-weight:bold"
    div_title.innerText =  data['title'][0] 

    var div_subtitle = document.createElement('div')
    div_subtitle.style = ";width:100%;font-size:20px ;font-family:'宋体';color:black;"
    div_subtitle.innerText =  data['subtitle'][0]  

    var div_text = document.createElement('div')
    div_text.style = "text-indent:2em;width:100% ;line-height:50px;;font-size:20px"
    div_text.innerText =  data['text'] 

    div_content_main.append(div_title)
    div_content_main.append(div_subtitle)
    div_content_main.append(div_text)

    div_main.append(div_blank)
    div_main.append(div_content_main)

    $('#product_view').append(div_main)


}
function list_div(data) {
    console.log(data['type'])
    var div_main = document.createElement('div')
    div_main.style = "display:flex; justify-content:center align-items:center;height:auto;width:100%;background-color:white;font-size:20px ;font-family:'宋体';color:black;line-height:50px;"

    var div_blank = document.createElement('div')
    div_blank.style = "width: 10%;"

    var div_content_main = document.createElement('div')
    div_content_main.style = "width: 80%;"

    var div_content_p = document.createElement('div')
    div_content_p.style = "display:inline-block;text-indent:1.5em;width:100%;height: 40% ;line-height:50px;;font-size:20px;"

    var p_text = document.createElement('p')
    p_text.style = ";width:100%;font-size:20px ;font-family:'宋体';color:black;;font-weight:bold"
    p_text.innerText =  data['title'][0]

    var p_text1 = document.createElement('p')
    p_text1.style = "float:left;line-height:40px;"
    p_text1.innerText =  data['text_list1']

    var p_text2 = document.createElement('p')
    p_text2.style = "float:left;line-height:40px;"
    p_text2.innerText =  data['text_list2']

    var p_text3 = document.createElement('p')
    p_text3.style = "float:left;line-height:40px;"
    p_text3.innerText =  data['text_list3']

    div_content_p.append(p_text1,p_text2,p_text3)

    div_content_main.append(p_text,div_content_p)

    div_main.append(div_blank)
    div_main.append(div_content_main)

    $('#product_view').append(div_main)

}
function png_div(data) {
    console.log(data['type'])

}
function foot_div(data) {
    console.log(data['type'])
    var div_main = document.createElement('div')
    div_main.style = "display:flex;height:auto;width:100%;background-color:white;font-size:20px ;font-family:'宋体';color:black;"

    var div_blank = document.createElement('div')
    div_blank.style = "width: 10%;"

    var div_content_main = document.createElement('div')
    div_content_main.style = "width: 80%;"

    var div_png_main = document.createElement('div')
    div_png_main.style = "width:100% ;text-align: center;"

    var div_png_1 = document.createElement('div')
    div_png_1.style = "display: inline-flex;width:25%;height:25%"
    var img_1 = document.createElement('img')
    img_1.style = ";width:100%;height:100%"
    img_1.src = "static/" + data['png1']
    div_png_1.append(img_1)

    var div_png_2 = document.createElement('div')
    div_png_2.style = "display: inline-flex;width:25%;height:25%"
    var img_2 = document.createElement('img')
    img_2.style = ";width:100%;height:100%"
    img_2.src = "static/" + data['png2']
    div_png_2.append(img_2)

    var div_png_3 = document.createElement('div')
    div_png_3.style = "display: inline-flex;width:25%;height:25%"
    var img_3 = document.createElement('img')
    img_3.style = ";width:100%;height:100%"
    img_3.src = "static/" + data['png3']
    div_png_3.append(img_3)

    div_png_main.append(div_png_1,div_png_2,div_png_3)

    var div_title_main = document.createElement('div')
    div_title_main.style = "display: flex;justify-content: center;text-align: center;height:5%;width:100%;"

    var div_title_1 = document.createElement('div')
    div_title_1.style = "display: inline-flex;width:25%;height:100%;text-align: center;"
    div_title_1.innerText = "  " + data['title1']

    var div_title_2 = document.createElement('div')
    div_title_2.style = "display: inline-flex;width:25%;height:100%;text-align: center;"
    div_title_2.innerText = "   " + data['title2']

    var div_title_3 = document.createElement('div')
    div_title_3.style = "display: inline-flex;width:25%;height:100%;text-align: center;"
    div_title_3.innerText = '   ' + data['title3']
    div_title_main.append(div_title_1,div_title_2,div_title_3)   

    var div_blank2 = document.createElement('div')
    div_blank2.style = "height:5%;width:100%;"

    var div_black_line = document.createElement('div')
    div_black_line.style = "height:1%;width:100%;border: 2px solid rgb(12, 12, 12);"

    var div_main_p = document.createElement('div')
    div_main_p.style = "width:100%;height:auto"

    var main_p1 = document.createElement('p')
    main_p1.style = "line-height:20px;"
    main_p1.innerText = "呈送领导:" +data['service_name']

    var main_p2 = document.createElement('p')
    main_p2.style = "line-height:20px;"
    main_p2.innerText = "抄送单位:" +data['service_unity']

    var main_p3 = document.createElement('p')
    main_p3.style = "line-height:20px;"
    main_p3.innerText = "抄送领导:" +data['recive_unity']

    div_main_p.append(main_p1,main_p2,main_p3)

    div_content_main.append(div_png_main,div_title_main,div_blank2,div_black_line,div_main_p)


    div_main.append(div_blank)
    div_main.append(div_content_main)

    $('#product_view').append(div_main)












}
// $('#save_and_vue').click(function () {
//     var list_data_all = [
//         {  //# 文件头
//             "type": "一周天气预测",
//             "company": "台州市气象台",
//             "writer": "翁之梅",
//             "putter": "高丽",
//             "index": 15,
//             "time": 'Wed Mar 01 2022 19:18:00 GMT+0800 (中国标准时间)'
//         },
//         {  //# 文档内容
//             "type": "段落",
//             "title": ["本周后期强冷空气影响，全市有严重冰冻", 1, 2, 3],
//             "subtitle": ["短信内容", 1, 2, 3],
//             "text": "本周前期天气阴沉，周后期强冷空气影响转晴冷天气，全市有严重冰冻。今天、明天阴有分散性小雨，最高气温12～14度；后天多云到阴，夜里～周四受强冷空气影响有明显降温、大风和弱雨雪过程，预计日平均气温过程降温幅度可达8～10度，后天夜里部分地区有小雨夹雪或雪，山区有小雪；周五～周日天气晴冷，其中周五、周六早晨北部地区降至零下6～零下8度，山区零下9～零下12度，其它地区零下4～零下6度，全市有严重冰冻，须注意防寒保暖。另外，周二沿海风力8～9级，周三夜里～周四沿海和内陆分别有8～10级和6～8级偏北大风。"
//         },
//         {  //# 文档列表
//             "type": "列表",
//             "title": ["台州市主城区具体天气预报如下：", 1, 2, 3],
//             "subtitle": ["短信内容", 1, 2, 3],
//             "text_list1": "4日（周一）：阴，部分地区有时有小雨              9～14度",
//             "text_list2": "4日（周一）：阴，部分地区有时有小雨        9～14度",
//             "text_list3": "4日（周一）：阴，部分地区有时有小雨        9～14度",
//             "text_list4": "4日（周一）：阴，部分地区有时有小雨        9～14度",
//             "text_list5": "4日（周一）：阴，部分地区有时有小雨        9～14度",
//             "text_list6": "4日（周一）：阴，部分地区有时有小雨        9～14度",

//         },
//         {  //# 图片列表
//             "type": "单图",
//             "title": "台州市气象台",
//             "png": 'src/pre_test.png'
//         },
//         {
//             "type": "文件尾",
//             "title1": "微信宣传",
//             "png1": 'src/二维码.png',
//             "title2": "气象台宣传",
//             "png2": 'src/二维码.png',
//             "title3": "浙政钉宣传",
//             "png3": 'src/二维码.png',
//             "service_name": '宋伟2、李渊、小王、张三',
//             "service_unity": '气象台、玉环市气象局、水利局、气象台、玉环市气象局、水利局、气象台、玉环市气象局、水利局、玉环市气象局、水利局、气象台、玉环市气象局、水利局、玉环市气象局、水利局、气象台、玉环市气象局、水利局、玉环市气象局、水利局、气象台、玉环市气象局、水利局',
//             "recive_unity": '气象台、玉环市气象局、水利局、玉环市气象局、水利局'
//         }

//     ]
//     $('#product_view').html(" ")


//     for (const i of list_data_all) {
//         if (i['type'] == '一周天气预测') {
//             head_div(i)
//         }
//         else if (i['type'] == '单图') {
//             // console.log(i)
//             png_div(i)
//         }
//         else if (i['type'] == '列表') {
//             // console.log(i)
//             list_div(i)
//         }
//         else if (i['type'] == '文件尾') {
//             // console.log(i)
//             foot_div(i)            
//         }
//         else if (i['type'] == '段落') {
//             // console.log(i)
//             text_div(i)
//         }

//     }

// })