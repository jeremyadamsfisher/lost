import React, {Component} from 'react'
import './Annotation.scss';
import * as modes from '../types/modes'
import * as transform from '../utils/transform'
import * as canvasActions from '../types/canvasActions'
import InfSelectionArea from './InfSelectionArea'
import Node from './Node'

class BBox extends Component{

    /*************
     * LIFECYCLE *
    **************/
    constructor(props){
        super(props)
        this.state = {
            anno: undefined,
            // selectedNode: undefined,
            // mode: modes.VIEW
        }
    }

    componentDidMount(prevProps){
        // console.log('Component mounted', this.props.data.id)
        if (this.props.anno.initMode === modes.CREATE){
            console.log('in Create Pos')
            const data = this.props.anno.data[0]
            const newAnno = {
                ...this.props.anno,
                data: [
                    {x: data.x, y: data.y},
                    {x: data.x+1, y: data.y},
                    {x: data.x+1, y: data.y+1},
                    {x: data.x, y: data.y+1}
                ],
                selectedNode: 2
            }
            this.setState({
                anno: newAnno
            })
            this.performedAction(newAnno, canvasActions.ANNO_START_CREATING)
            // this.setMode(modes.CREATE, 2)
            
        } else {
            this.setState({anno: {...this.props.anno}})
        }
    }

    componentDidUpdate(prevProps){
        if (prevProps.anno !== this.props.anno){
            this.setState({anno: {...this.props.anno}})
        }
    }

    /*************
    * EVENTS    *
    **************/
    onNodeMouseMove(e, idx){
        switch (this.state.anno.initMode){
            case modes.CREATE:
            case modes.EDIT:
                const idxMinus = idx - 1 < 0 ? 3 : idx -1
                const idxPlus = idx + 1 > 3 ? 0 : idx +1
                let newAnnoData = [...this.state.anno.data]
                const movementX = e.movementX / this.props.svg.scale
                const movementY = e.movementY / this.props.svg.scale
                if (idx % 2 === 0){
                    newAnnoData[idxMinus].x += movementX
                    newAnnoData[idx].x += movementX
                    newAnnoData[idx].y += movementY
                    newAnnoData[idxPlus].y += movementY
                } else {
                    newAnnoData[idxMinus].y += movementY
                    newAnnoData[idx].x += movementX
                    newAnnoData[idx].y += movementY
                    newAnnoData[idxPlus].x += movementX
                }
                this.setState({
                    anno: {
                        ...this.state.anno,
                        data: newAnnoData
                    }
                })
                break
            default:
                break
        }
    }

    onNodeMouseDown(e,idx){
        switch(this.state.anno.initMode){
            case modes.VIEW:
                if (e.button === 0){
                    console.log('Node mouse Down', idx)
                    // this.setMode(modes.EDIT, idx)
                    this.requestModeChange(
                        {...this.state.anno, selectedNode:idx}, 
                        modes.EDIT
                    )
                    // this.setState({selectedNode: idx})
                }
                break
        }
    }

    onNodeMouseUp(e, idx){
        switch(this.state.anno.initMode){
            case modes.EDIT:
                if (e.button === 0){
                    // this.setMode(modes.VIEW)
                    this.requestModeChange(this.state.anno, modes.VIEW)
                    this.performedAction(this.state.anno, canvasActions.ANNO_EDITED)
                }
                break
            case modes.CREATE:
                if (e.button === 2){
                    // this.setMode(modes.VIEW)
                    console.log('BBOX: hist Created', this.state.anno)
                    this.requestModeChange(this.state.anno, modes.VIEW)
                    this.performedAction(this.state.anno, canvasActions.ANNO_CREATED)
                }
        }
    }

    /**************
    * ANNO EVENTS *
    ***************/
    onMouseMove(e){
        switch (this.state.anno.initMode){
            case modes.MOVE:
                this.move(
                    e.movementX/this.props.svg.scale, 
                    e.movementY/this.props.svg.scale
                )
                break
            default:
                break
        }
    }

    onMouseUp(e){
        switch (this.state.anno.initMode){
            case modes.MOVE:
                if (e.button === 0){
                    this.requestModeChange(this.state.anno, modes.VIEW)
                    this.performedAction(this.state.anno, canvasActions.ANNO_MOVED)
                    // this.setMode(modes.VIEW)
                }
                break
            default:
                break
        }
    }

    onMouseDown(e){
        switch (this.state.anno.initMode){
            case modes.VIEW:
                if (e.button === 0){
                    if (this.props.isSelected){
                        this.requestModeChange(this.state.anno, modes.MOVE)
                        // this.setMode(modes.MOVE)
                    }
                }
                break
        }
    }
    /*************
    *  LOGIC     *
    **************/
    requestModeChange(anno, mode){
        this.props.onModeChangeRequest(anno, mode)
    }

    performedAction(anno, pAction){
        if (this.props.onAction){
            this.props.onAction(anno, pAction)
        }
    }
    // setMode(mode, nodeIdx=undefined){
    //     if (this.state.mode !== mode){
    //         switch (mode){
    //             case modes.MOVE:
    //             case modes.EDIT:
    //                 if (this.props.allowedToEdit){
    //                     if (this.props.onModeChange){
    //                         this.props.onModeChange(mode, this.state.mode)
    //                     }
    //                     this.setState({
    //                         mode: mode,
    //                         selectedNode: nodeIdx
    //                     })
    //                 }
    //                 break
    //             default:
    //                 if (this.props.onModeChange){
    //                     this.props.onModeChange(mode, this.state.mode)
    //                 }
    //                 this.setState({
    //                     mode: mode,
    //                     selectedNode: nodeIdx
    //                 })
    //                 break
    //         }
    //     }
    // }
    toPolygonStr(data){
        return data.map( (e => {
            return `${e.x},${e.y}`
        })).join(' ')
        
    }

    move(movementX, movementY){
        this.setState({
            anno : {
                ...this.state.anno,
                data: transform.move(this.state.anno.data, movementX, movementY)
            }
        })
    }

    /*************
     * RENDERING *
    **************/

    renderPolygon(){
        switch(this.state.anno.initMode){
            case modes.MOVE:
            case modes.EDIT:
            case modes.VIEW:
            case modes.CREATE:
            case modes.EDIT_LABEL:
                return <polygon 
                            points={this.toPolygonStr(this.state.anno.data)}
                            fill='none' stroke="purple" 
                            style={this.props.style}
                            className={this.props.className}
                            onMouseDown={e => this.onMouseDown(e)}
                            onMouseUp={e => this.onMouseUp(e)}
                        />
            default:
                return null 
        }
    }

    renderNodes(){
        if (!this.props.isSelected) return null 
        switch(this.state.anno.initMode){
            case modes.MOVE:
            case modes.EDIT_LABEL:
                return null
            case modes.EDIT:
            case modes.CREATE:
                return <Node anno={this.state.anno.data}
                            key={this.state.anno.selectedNode}
                            idx={this.state.anno.selectedNode} 
                            style={this.props.style}
                            className={this.props.className} 
                            isSelected={this.props.isSelected}
                            mode={this.state.anno.initMode}
                            svg={this.props.svg}
                            onMouseDown={(e, idx) => this.onNodeMouseDown(e,idx)}
                            onMouseUp={(e, idx) => this.onNodeMouseUp(e, idx)}
                            onMouseMove={(e, idx) => this.onNodeMouseMove(e, idx)}
                        />
            default:
                console.log('BBOX: renderNodes default', this.state.anno)
                return this.state.anno.data.map((e, idx) => {
                    return <Node anno={this.state.anno.data} idx={idx} 
                        key={idx} style={this.props.style}
                        className={this.props.className} 
                        isSelected={this.props.isSelected}
                        mode={this.state.anno.initMode}
                        svg={this.props.svg}
                        onMouseDown={(e, idx) => this.onNodeMouseDown(e,idx)}
                        onMouseUp={(e, idx) => this.onNodeMouseUp(e, idx)}
                        />
                })
        }
    }

    renderInfSelectionArea(){
        switch (this.state.anno.initMode){
            case modes.MOVE:
                return <InfSelectionArea enable={true} 
                        svg={this.props.svg}
                    />
            default:
                return null
        }
    }

    render(){
        if (this.state.anno){
            console.log('BBOX: annoChangeMode Render', this.state.anno)
            return (
            <g
                onMouseMove={e => this.onMouseMove(e)}
                onMouseUp={e => this.onMouseUp(e)}
                onMouseDown={e => this.onMouseDown(e)}
            >
                {this.renderPolygon()}
                {this.renderNodes()}
                {this.renderInfSelectionArea()}
            </g>)
        } else {
            return null
        }
    }

}

export default BBox;