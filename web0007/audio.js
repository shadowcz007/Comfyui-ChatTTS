import { app } from '../../../scripts/app.js'
import { $el, ComfyDialog } from '../../../scripts/ui.js'

// 扩展原型链上的 close 方法
const originalClose = ComfyDialog.prototype.close
ComfyDialog.prototype.close = function () {
  // console.log('#ComfyDialog', 111111) // 新增的console日志
  originalClose.call(this) // 调用原始的 close 方法

  const nodes = app.graph.findNodesByType('EditMask')

  for (const node of nodes) {
    const image_update = node.widgets.filter(w => w.name == 'image_update')[0]
    // console.log(image_update)
    image_update.value = { images: node.images }
  }
}

function get_position_style (ctx, widget_width, y, node_height) {
  const MARGIN = 4 // the margin around the html element

  /* Create a transform that deals with all the scrolling and zooming */
  const elRect = ctx.canvas.getBoundingClientRect()
  const transform = new DOMMatrix()
    .scaleSelf(
      elRect.width / ctx.canvas.width,
      elRect.height / ctx.canvas.height
    )
    .multiplySelf(ctx.getTransform())
    .translateSelf(MARGIN, MARGIN + y)

  return {
    transformOrigin: '0 0',
    transform: transform,
    left: `0`,
    top: '0',
    cursor: 'pointer',
    position: 'absolute',
    maxWidth: `${widget_width - MARGIN * 2}px`,
    // maxHeight: `${node_height - MARGIN * 2}px`, // we're assuming we have the whole height of the node
    width: `${widget_width - MARGIN * 2}px`,
    // height: `${node_height * 0.3 - MARGIN * 2}px`,
    // background: '#EEEEEE',
    display: 'flex',
    flexDirection: 'column',
    // alignItems: 'center',
    justifyContent: 'space-around'
  }
}

app.registerExtension({
  name: 'Mixlab_test.editMask',
  async getCustomWidgets (app) {
    return {
      IMAGE_ (node, inputName, inputData, app) {
        const widget = {
          type: inputData[0], // the type, CHEESE
          name: inputName, // the name, slice
          size: [128, 88], // a default size
          draw (ctx, node, width, y) {},
          callback: e => console.log(e),
          computeSize (...args) {
            return [128, 88] // a method to compute the current size of the widget
          },
          async serializeValue (nodeId, widgetIndex) {
            // node=app.graph.getNodeById(node.id)
            if (node.images) return { images: node.images }
            return {}
          }
        }
        node.addCustomWidget(widget)
        return widget
      }
    }
  },

  async beforeRegisterNodeDef (nodeType, nodeData, app) {
    // 节点创建之后的事件回调
    const orig_nodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      orig_nodeCreated?.apply(this, arguments)
    }
  },
  async loadedGraphNode (node, app) {
    if (node.type === 'EditMask') {
      let image_update = node.widgets.filter(w => w.name === 'image_update')[0]
      console.log('#image_update', image_update)
      // let img = new Image()
      // img.src = image_update
      // node.images = []
    }
  }
})

app.registerExtension({
  name: 'chattts.AudioPlayNode_TEST',
  // 节点被初次创建的时候，将会xxxx
  async beforeRegisterNodeDef (nodeType, nodeData, app) {
    if (nodeType.comfyClass == 'AudioPlayNode_TEST') {
      console.log('#AudioPlayNode_TEST000000999999', nodeData)

      // 节点创建之后的事件回调
      const orig_nodeCreated = nodeType.prototype.onNodeCreated
      nodeType.prototype.onNodeCreated = function () {
        orig_nodeCreated?.apply(this, arguments)
      }

      // 节点执行之后的事件回调
      const onExecuted = nodeType.prototype.onExecuted
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments)
        console.log('#节点执行之后的事件回调', message)

        const { audio1 } = message // 熟悉js

        let filename = audio1[0].filename // 数组
        console.log('#filename', filename)
        //传递不同的变量，怎么获取

        const widget = {
          type: 'div',
          name: 'filename',
          draw (ctx, node, widget_width, y, widget_height) {
            Object.assign(
              this.div.style,
              get_position_style(ctx, widget_width, y, node.size[1])
            )
          }
        }

        console.log('#widget', widget)

        widget.div = $el('div', {}) // 创建一个html的元素

        document.body.appendChild(widget.div)

        widget.div //可以怎么修改，

        // 创建按钮元素
        const button = document.createElement('button')
        button.innerText = filename // 按钮文字

        // 设置按钮样式
        button.style.display = 'inline-block'
        button.style.backgroundColor = 'black'
        button.style.color = 'white'
        button.style.padding = '10px 20px'
        button.style.border = 'none'
        button.style.cursor = 'pointer'
        button.style.fontSize = '16px'

        // 创建分享图标元素
        const shareIcon = document.createElement('div')

        // 设置分享图标样式
        shareIcon.style.display = 'inline-block'
        shareIcon.style.width = '24px'
        shareIcon.style.height = '24px'
        shareIcon.style.backgroundColor = 'red'
        shareIcon.style.marginLeft = '10px'
        shareIcon.style.verticalAlign = 'middle'

        // 将分享图标添加到按钮后面
        button.appendChild(shareIcon)

        // 将按钮添加到容器中
        //  const container = document.getElementById('container');
        widget.div.appendChild(button)

        // widget.div.innerText = filename //把div里的正文改成filename

        //添加控件到节点里
        this.addCustomWidget(widget)
      }
    }

    if (nodeType.comfyClass == 'EditMask') {
      console.log('#EditMask', nodeData)
    }
  },
  async loadedGraphNode (node, app) {}
})
