var map = echarts.init(document.getElementById('map_tool1'),'light');
echarts.registerMap('taizhou',taizhou,{});
map.setOption({
          title:{
      text:'',
      textStyle:{fontSize:50},
      },
      tooltip: {
          trigger: 'item',
            // formatter: '{c}(m/s)'
            formatter: function (params) {
              // do some thing
              return  params.value[2] +params.name
           }
      },
      geo: {
                      map: 'taizhou',
                      roam: true,
                      // tooltip:{borderColor = '#333'}
                      itemStyle:{
                        color:'white'
                          // areaColor = '#333',
                      }


                  },
      
          // label: {
          //             show: true,
          //             color: "rgba(50, 44, 44, 1)",
          //             formatter: 'K8505'
          //         },
      series: [{
                      name: '确诊数量',
                      // type: 'effectScatter',
                      type: 'scatter',
                      // symble:"triangle",
                      // symbol: "roundRect",
                      // symbolSize:50,
                      coordinateSystem: 'geo'
                      // itemStyle: {
                      //   color: function (params) {
                      //       var colorlist = ['black'];
                      //       return colorlist[params.dataIndex];
                      //   },
                      //   symbol: function (params) {
                      //     var symbollist = ['path://M10 10L60 10 60 20 20 20 20 40 60 40 60 50 20 50 20 100 10 100 10 10z'];
                      //     return symbollist[params.dataIndex];
                      // },
                    // },

                      // data:[{ name:"K8515",value: [121.0, 28.8, 123], url: "url_data",
                      // symbol:'path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z',symbolSize:15,symbolRotate:330}]

                  }]



  });

//  map.on('click', (params) => {
//     console.log('click', params);
//     // 执行方法
//   });

  map.on('click', function (params) {
	let componentType = params.componentType;   // geo是地图图层
	if (componentType == "geo") {
		var offsetX = params.event.offsetX;
		var offsetY = params.event.offsetY;
		var zuobiao = map.convertFromPixel('geo', [offsetX, offsetY]); // 转换成坐标
    // console.log(zuobiao,"数据",params.name) 
    $('#h4_village').html("乡镇名称:"+" " + params.name)
    $('#h4_lon').html("经度:"+" " + zuobiao[0].toFixed(2))
    $('#h4_lat').html("纬度:"+" " + zuobiao[1].toFixed(2))

	}
});

