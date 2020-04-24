var gameBoard = document.querySelector('#board');
gameBoard.addEventListener('click', getColumnId)

function sendActionToFlask(column_id){
    var urlToPost = '/' + column_id;
    $.ajax({
        url: urlToPost,
        type: 'GET',
        // Get a json here from flask update_grid() and modify the document in success function
        success: function(result, status, xhr){
            console.log("Sent");
            },
        error: function(xhr, status, error) {
            console.log(xhr.status + ": " + xhr.responseText);
        }
    });
}

function getColumnId(e){
    if (e.target.tagName !== 'BUTTON') return;

    var targetCell = e.target.parentElement;
    var targetRow = targetCell.parentElement;
    var targetRowCells = [...targetRow.children];

    var column_id = targetRowCells.indexOf(targetCell);

    sendActionToFlask(column_id)
}
