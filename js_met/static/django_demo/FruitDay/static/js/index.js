$(function (){
	//文档加载完毕后执行
	/*1. 下拉菜单点击传值*/
	$(".address li").each(function (){ //遍历元素
		//为元素绑定单击事件
		$(this).click(function (){
			//点击事件触发后执行的操作-传值
			$(".address span").html($(this).html());
		})
	})
	/*2. 轮播图*/
	//eq(index) 获取指定下标的元素
	var imgIndex = 0;
	var timerID = setInterval(autoPlay,1000);
	function autoPlay(){
		//隐藏图片
		$("#banner img").each(function (){
			$(this).css("display","none");
		})
		//更新下标 ++n + n
		imgIndex = ++ imgIndex == $("#banner img").length ? 0 : imgIndex;

		//显示一张图片
		$("#banner img").eq(imgIndex).css("display","block");
		
		//图片索引
		$("#banner li").each(function (){
			$(this).css("background","gray");
		})
		//跟随图片显示背景色
		$("#banner li").eq(imgIndex).css("background","red");

	}
	//鼠标移入,停止定时器
	$("#banner").bind("mouseover",function (){
		clearInterval(timerID);
	})
	//鼠标移出,继续轮播
	$("#banner").mouseout(function (){
		timerID = setInterval(autoPlay,1000);
	})
})