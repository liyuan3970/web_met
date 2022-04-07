//饼图数据
pieData = [[{ "name": "A系统", "value": 2 }, { "name": "B系统", "value": 4 }, { "name": "C系统", "value": 3 }, { "name": "D系统", "value": 3 }, { "name": "E系统", "value": 7 }, { "name": "F系统", "value": 3 }, { "name": "G系统", "value": 3 }, { "name": "H系统", "value": 3 }], ["A系统", "B系统", "C系统", "D系统", "E系统", "F系统", "G系统", "H系统"]];

//折线图数据
yqyData = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [600, 1150, 400, 850, 600, 650, 100, 1150, 100, 700, 600, 800, 500]]

//---------------------------------------------关系图数据获取:start，有点多，可先忽略查看----------------------------------------------------------

//用java代码：自动生成关系图数据（因为前期，没有理解需求——_——:全是自己用java自动生成的json数据）
relatData = [{ "node": "系统节点1", "endpoint": ["系统节点3"], "service": ["java.local.name_1", "java.local.name_11"] }, { "node": "系统节点2", "endpoint": ["系统节点3", "系统节点1"], "service": ["java.local.name_2", "java.local.name_22"] }, { "node": "系统节点3", "endpoint": ["系统节点4", "系统节点2"], "service": ["java.local.name_3", "java.local.name_33"] }, { "node": "系统节点4", "endpoint": ["系统节点3"], "service": ["java.local.name_4", "java.local.name_44"] }, { "node": "系统节点5", "endpoint": ["系统节点2"], "service": ["java.local.name_5", "java.local.name_55"] }, { "node": "系统节点6", "endpoint": ["系统节点3"], "service": ["java.local.name_6", "java.local.name_66"] }, { "node": "系统节点7", "endpoint": ["系统节点2"], "service": ["java.local.name_7", "java.local.name_77"] }, { "node": "系统节点8", "endpoint": ["系统节点2"], "service": ["java.local.name_8", "java.local.name_88"] }];

//获取节点数据
function get_nodes(relatData) {//官方的方法改造了一下=自定义生成：关系图中节点位置
    var nodes = [];
    var tmp_nodes = [];
    var x1 = 1200;
    var y1 = 100;

    for (var nodes_i in relatData) {
        //x,y数据归位
        x1 = 5;
        y1 = 5;
        //三个节点为一排，之字形增加
        x1 = x1 + 10 * (nodes_i % 3);//保持0，1，2
        y1 = y1 + 10 * Math.floor(nodes_i / 3);//向下取整

        console.log("x1=" + x1 + " y1=" + y1);
        console.log("x1=" + (nodes_i % 3) + " y1=" + (Math.floor(nodes_i / 3)));
        tmp_nodes.push(relatData[nodes_i].node);
        nodes.push(
            {

                'name': relatData[nodes_i].node,
                'value': [x1, y1]//4.通过x,y来确定关系图的节点位置
            }
        );
    }
    return nodes;
}

//获取节点数据关系
function get_links(relatData) {
    var links = [];
    for (var nodes_i in relatData) {
        var node = relatData[nodes_i].node;
        var endpoint = relatData[nodes_i].endpoint;
        var service = relatData[nodes_i].service;
        // console.log(service);
        for (var service_i in endpoint) {
            links.push({
                'source': node,
                'target': endpoint[service_i],
                'label': {
                    'normal': {
                        'show': false,
                        'textStyle': {
                            'fontSize': 5
                        },
                        'formatter': service
                    }
                },
                'lineStyle': {
                    'normal': {
                        'curveness': 0.1
                    }
                }
            })
        }
    }
    for (var i = 0, len1 = links.length; i < len1; i++) {
        for (var j = i, len2 = len1 - 1; j < len2; j++) {
            if (links[i].source == links[j].target) {
                links[j].lineStyle.normal.curveness = -0.1;
            }
        }
    }
    // console.log(links);
    return links;
}
//------------------------------------------------------关系图获取数据:end-----------------------------------------------------------------

//按老大要求：多个图表数据整合成一个option
//【关键点】：1.用最少的代码显示出一个图（eg:饼图只要一个series就可以显示出来）
//2.确定每个图表的位置，占的区域（eg:饼图通过配置center来确定中心位置，radius确定饼图的大小）。
function oneForAll() {

    //我选折线图作为基础option
    var yqyOption = {
        xAxis: [
            { data: yqyData[0], gridIndex: 0 },//折线图x轴数据赋值，指定坐标信息
            { gridIndex: 1, type: 'value' }
        ],
        yAxis: [
            {
                name: '交易量',
                splitLine: { show: false },
                gridIndex: 0
            },
            { gridIndex: 1, type: 'value' }
        ],

        grid: [//指定坐标轴位置，大小
            { x: '7%', y: '7%', width: '50%', height: '31%' },
            { x: '60%', bottom: '1%', height: '90%', width: '35%', contianLabel: true }//关系图位置
        ],

        series: [{
            type: 'line',
            xAxisIndex: 0,//指定折线图数据显示到：grid坐标系：0
            yAxisIndex: 0,
            showSymbol: false,
            data: yqyData[1]//折线图y轴数据赋值
        }]
    };

    //======饼图数据1=====
    var pieSeries1 = {
        name: '面积模式',
        type: 'pie',
        radius: [10, 80],
        center: ['18%', '75%'],//位置确定：左下角
        data: pieData[0]//饼图数据赋值
    };

    //======饼图数据2=====
    var pieSeries2 = {
        name: '面积模式',
        type: 'pie',
        radius: [10, 80],
        center: ['47%', '75%'],
        roseType: 'area',//zxj玫瑰饼图
        data: pieData[0]
    };

    //将饼图series添加到主series中
    yqyOption.series.push(pieSeries1);//完成了一个简单的折线图和饼图的数据合并
    yqyOption.series.push(pieSeries2);

    //关系图数据
    var relatSeries =
    {
        type: 'graph',
        // layout: 'circular',
        layout: 'force',//1.力引导图
        coordinateSystem: 'cartesian2d',//2.笛卡尔坐标系设置
        xAxisIndex: 1,//3.选取的第二个坐标系，为关系图数据位置，4.value值设置
        yAxisIndex: 1,
        focusNodeAdjacency: true,
        legendHoverLink: true,
        hoverAnimation: true,
        symbolSize: 30,
        edgeSymbolSize: 10,
        roam: true,
        symbol: "roundRect",
        label: {
            normal: {
                show: true,
            }
        },
        edgeSymbol: ['circle', 'arrow'],
        edgeSymbolSize: [4, 15],
        edgeLabel: {
            normal: {
                textStyle: {
                    fontSize: 20
                }
            }
        },
        data: get_nodes(relatData),//节点数据赋值
        links: get_links(relatData),

        lineStyle: {
            normal: {
                opacity: 0.9,
                width: 2,
                curveness: 0,
                type: 'dashed'
            }
        }
    };

    //将关系图添加到主series中
    yqyOption.series.push(relatSeries);

    return yqyOption;
}