from rest_framework.renderers import JSONRenderer


class CustomRenderer(JSONRenderer):
    def render(self, data, accepted_media_type=None, renderer_context=None):
        if renderer_context:
            response = renderer_context.get("response")
            if isinstance(data, dict):
                err_msg = data.pop("err_msg", None)
                if err_msg:
                    msg = err_msg
                    code = response.status_code
                    data = None
                else:
                    msg = "success"
                    code = response.status_code

                ret = {
                    'msg': msg,
                    'code': code,
                    'data': data,
                }
                return super().render(ret, accepted_media_type, renderer_context)
            else:
                return super().render(data, accepted_media_type, renderer_context)
