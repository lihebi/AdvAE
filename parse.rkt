#lang racket

;; some utilities written in rkt

(require json)

(define (parse-train-process-from-stdout)
  (define-values (new-exp collect-exp get-result)
    (let ([all '()]
          [current '()])
      (values
       (λ (name)
         (set! all (append all (list current)))
         ;; the key of jsexpr has to be a symbol, not string
         (set! current (list (string->symbol name))))
       (λ (x)
         (set! current (append current (list x))))
       (λ ()
         (let ([lst (filter (λ (x) (> (length x) 1))
                            (append all (list current)))])
           ;; change this list to a hash table
           (make-hash lst))))))
  (call-with-input-file "stdout.txt"
    (λ (in)
      ;; "====== Denoising training for saved_models/MNIST-mnistcnn-cnn3AE-C0_A2_0.5-AdvAE.hdf5 .."
      ;; "Trainng AdvAE .."
      ;; "Epoch 1/100"
      ;; "54000/54000 [==============================] - 117s 2ms/step - loss: 1.7462 - val_loss: 1.3441"
      ;; "{'advacc': 0.5018382352941176, 'acc': 0.8801804812834224, 'cnnacc': 0.9908088235294118, 'obliacc': 0.7984625668449198}"
      ;; "Epoch 2/100"
      ;; "Restoring model weights from the end of the best epoch"
      (let ([lines (port->lines in)])
        (for ([line lines])
          (cond
            [(string-prefix? line "====== Denoising training for")
             (new-exp (second (regexp-match #px"saved_models/(.*)\\.hdf5" line)))]
            [(string-prefix? line "Trainng AdvAE ..")]
            [(string-prefix? line "Epoch 1/100")]
            [(string-prefix? line "54000/54000 [==============================]")
             (collect-exp (string->number (second (regexp-match #px"val_loss: (\\d*\\.\\d*)" line))))]
            [(string-prefix? line "{'advacc': ")]
            [(string-prefix? line "Restoring model weights from the end of the best epoch")]
            [else #f])))))
  (get-result))

(module+ test
  (jsexpr? (parse-train-process-from-stdout))
  (call-with-output-file "images/train-process.json"
    #:exists 'replace
    (λ (out)
      (write-json (parse-train-process-from-stdout) out))))
