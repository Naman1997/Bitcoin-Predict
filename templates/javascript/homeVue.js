// http://127.0.0.1:8080/test/30/

var app = new Vue({
    el: '#app',
    data: {
        dict:{}
    },
    created: function(){
        this.getResult();
        this.con();
    },
    methods: {
        getResultRNN(time){
            axios.get('http://127.0.0.1:8080/test/'+time).then(response =>{
                var incoming = response.data;
                var len = incoming.length;
                var dict = {};

                for(i=0; i<incoming.length;i++){
                        dict[i] = incoming[i];
                }

                this.dict = dict;

                console.log(dict);

            });
        },
        con(){
            console.log("HEYYYYYYYYYYYYYY");
        },
    }
});