import { app } from '../../../scripts/app.js'

app.registerExtension({
  name: 'Mixlab.Chattts.LoadSpeaker',

  async beforeRegisterNodeDef (nodeType, nodeData, app) {
    if (nodeType.comfyClass == 'LoadSpeaker') {
     
      const onExecuted = nodeType.prototype.onExecuted
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments)
        if (message.text && message.text[0]) {
          this.title=message.text
        }
      }
    }
  }
})
