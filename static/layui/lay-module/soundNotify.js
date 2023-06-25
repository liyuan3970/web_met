/**
 * 文字转语音、音频、桌面通知
 */
layui.define(function (exports) {
    "use strict";
    const MOD_NAME = 'soundNotify';

    function SoundNotify() {}

    // 文字转语音
    SoundNotify.prototype.sound = function (text) {
        let utterance = new SpeechSynthesisUtterance(text);
        return window.speechSynthesis.speak(utterance);
    };

    // 音频
    SoundNotify.prototype.audio = function (url) {
        let audio = new Audio(url);
        audio.autoplay = true;
        audio.play();
    };

    // 桌面通知
    // 消息授权必须是https协议的，否则一直会是 denied
    SoundNotify.prototype.notify = function (title, options) {
        // 先检查浏览器是否支持
        if (!window.Notification) {
            throw new Error('浏览器不支持Notification');
        }

        return new Promise((resolve, reject) => {
            const map = {
                // 用户同意授权
                granted() {
                    // 显示通知
                    resolve(new Notification(title, options));
                },
                // 用户还未选择 可以询问用户是否同意发送通知
                default() {
                    Notification.requestPermission().then(permission => {
                        map[permission]();
                    }).catch(error => {
                        reject(error);
                    });
                },
                // 用户拒绝授权 不能显示通知
                denied() {
                    reject('用户拒绝授权');
                }
            }

            map[Notification.permission]();
        })
    };

    exports(MOD_NAME, new SoundNotify());
});