import panel as pn


class SwitchSetting(pn.layout.Row):
    def __init__(self, name, value=False):
        self.value = value
        
        def change_value(event):
            self.value = event
        
        switch = pn.widgets.Switch(name=name, value=value, width=40)
        text = pn.widgets.StaticText(name=name, width=140)
        pn.bind(change_value, switch, watch=True)
        
        super().__init__(text, switch, sizing_mode='fixed')

        