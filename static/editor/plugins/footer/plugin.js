// tinymce.PluginManager.add('footer', function(editor, url) {
//     var openDialog = function () {
//       return editor.windowManager.open({
//         title: '这里是弹窗标题',
//         body: {
//           type: 'panel',
//           items: [
//             {
//               type: 'input',
//               name: 'title',
//               label: 'Title'
//             }
//           ]
//         },
//         buttons: [
//           {
//             type: 'cancel',
//             text: 'Close'
//           },
//           {
//             type: 'submit',
//             text: 'Save',
//             primary: true
//           }
//         ],
//         onSubmit: function (api) {
//           var data = api.getData();
//           // 将输入框内容插入到内容区光标位置
//           editor.insertContent('插入的文字是: ' + data.title);
//           api.close();
//         }
//       });
//     };
    
//     // 注册一个工具栏按钮名称
//     editor.ui.registry.addButton('footer', {
//       text: '工具栏按钮名',
//       onAction: function () {
//         openDialog();
//       }
//     });
  
//     // 注册一个菜单项名称 menu/menubar
//     editor.ui.registry.addMenuItem('footer', {
//       text: 'Example菜单名',
//       onAction: function() {
//         openDialog();
//       }
//     });
  
//     return {
//       getMetadata: function () {
//         return  {
//           //插件名和链接会显示在“帮助”→“插件”→“已安装的插件”中
//           name: "Example plugin",//插件名称
//           url: "http://exampleplugindocsurl.com", //作者网址
//         };
//       }
//     };
//   });