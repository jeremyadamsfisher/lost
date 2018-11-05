// @refactor: rename bba to sia in annotask

import "./app.styles.scss"
import { NodeTemplate } from "l3p-core"

import * as imageView from "./components/image/imageView"
import * as propertiesView from "./components/properties/propertiesView"
import * as toolbarView from "./components/toolbar/toolbarView"
import * as controlsView from "./components/controls/controlsView"

export const html = new NodeTemplate(`
    <div id="sia-content-wrapper">
        <div id="sia-app-hide-plane">
            <h4 class="panel-body">loading... <i class="fa fa-spinner load-icon" aria-hidden="true"></i></h4>
        </div>
    </div>
`)
// append views (would be nice to be able to directly embed nodetemplates)
html.root.appendChild(toolbarView.html.fragment)
html.root.appendChild(propertiesView.html.fragment)
html.root.appendChild(imageView.html.fragment)
document.getElementById("sia-user-manual-content").appendChild(controlsView.html.fragment)
document.getElementById("sia-drawer-content").appendChild(html.fragment)

export function show(){
    document.getElementById("bba-progress-bar").style.display = "block"
    html.ids["sia-app-hide-plane"].style.display = "none"
}
export function hide(){
    document.getElementById("bba-progress-bar").style.display = "none"
    html.ids["sia-app-hide-plane"].style.display = "block"
}
