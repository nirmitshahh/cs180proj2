    import cv2
    import numpy as np
    import scipy
    from skimage import io
    import matplotlib.pyplot as plt
    from align_image_code import align_images



    def convolutionfourloops(image, filter):
        x, y = np.shape(image)
        fx, fy = np.shape(filter)
        
        px = fx // 2
        py = fy // 2

        padded = np.zeros((x + (2 * px), y + (2 * py)))
        padded[px: x + px, py: y + py] = image

        out = np.zeros((x, y))

        filter = filter[::-1, ::-1]

        for i in range(x):
            for j in range(y):
                dot = 0
                for fi in range(fx):
                    for fj in range(fy):
                        dot += padded[i + fi][j + fj] * filter[fi][fj]

                out[i][j] = dot

        return out


    def convolutiontwoloops(image, filter):
        x, y = np.shape(image)
        fx, fy = np.shape(filter)
        
        px = fx // 2
        py = fy // 2

        padded = np.zeros((x + (2 * px), y + (2 * py)))
        padded[px: x + px, py: y + py] = image

        out = np.zeros((x, y))

        filter = filter[::-1, ::-1]

        for i in range(x):
            for j in range(y):
                
                dot = np.sum(filter * padded[i:i + fx, j:j + fy])

                out[i][j] = dot

        return out


    def partonepointone():

        image = io.imread("1.1image.jpg", as_gray=True)

        boxfilter = np.ones((9,9)) / (9 * 9)
        dxfilter = np.array([[1, 0, -1]])
        dyfilter = np.array([[1], [0], [-1]])

        print("started")
        four_box = convolutionfourloops(image, boxfilter)
        print("ended")
        print("started")
        four_dx = convolutionfourloops(image, dxfilter)
        print("ended")
        print("started")
        four_dy = convolutionfourloops(image, dyfilter)

        two_box = convolutiontwoloops(image, boxfilter)
        two_dx = convolutiontwoloops(image, dxfilter)
        two_dy = convolutiontwoloops(image, dyfilter)

        scipy_box = scipy.signal.convolve2d(image, boxfilter, mode="same")

        scipy_dx = scipy.signal.convolve2d(image, dxfilter, mode="same")

        scipy_dy = scipy.signal.convolve2d(image, dyfilter, mode="same")
        plt.imsave("original_gray.jpg", image, cmap="gray")

        plt.imsave("four_box.jpg", four_box, cmap="gray")
        plt.imsave("four_dx.jpg", four_dx, cmap="gray")
        plt.imsave("four_dy.jpg", four_dy, cmap="gray")

        plt.imsave("two_box.jpg", two_box, cmap="gray")
        plt.imsave("two_dx.jpg", two_dx, cmap="gray")
        plt.imsave("two_dy.jpg", two_dy, cmap="gray")

        plt.imsave("scipy_box.jpg", scipy_box, cmap="gray")
        plt.imsave("scipy_dx.jpg", scipy_dx, cmap="gray")
        plt.imsave("scipy_dy.jpg", scipy_dy, cmap="gray")



    def partonepointtwo():
        image = io.imread("cameraman.png", as_gray=True)

        dxfilter = np.array([[1, 0, -1]])
        dyfilter = np.array([[1], [0], [-1]])

        dx = scipy.signal.convolve2d(image, dxfilter, mode="same")
        dy = scipy.signal.convolve2d(image, dyfilter, mode="same")

        gradientmagnitude = np.sqrt(dx**2 + dy**2)

        print(np.min(gradientmagnitude))
        print(np.max(gradientmagnitude))
        print(np.median(gradientmagnitude))
        print("90%:", np.percentile(gradientmagnitude, 90))


        binarized = gradientmagnitude > 0.3

        plt.imsave("partonepointtwoimages/dx.png", dx, cmap="gray")
        plt.imsave("partonepointtwoimages/dy.png", dy, cmap="gray")
        plt.imsave("partonepointtwoimages/gradient_magnitude.png", gradientmagnitude, cmap="gray")
        plt.imsave("partonepointtwoimages/edges.png", binarized, cmap="gray")


    def partonepointthree():
        image = io.imread("cameraman.png", as_gray=True)

        dxfilter = np.array([[1, 0, -1]])
        dyfilter = np.array([[1], [0], [-1]])

        oned = cv2.getGaussianKernel(9, 0)
        twod = oned @ oned.T

        blur = scipy.signal.convolve2d(image, twod, mode = "same")

        dx = scipy.signal.convolve2d(blur, dxfilter, mode = "same")
        dy = scipy.signal.convolve2d(blur, dyfilter, mode = "same")

        blurgradientmagnitude = np.sqrt(dx**2 + dy**2)

        blurbinarized = blurgradientmagnitude > 0.1

        dogx = scipy.signal.convolve2d(twod, dxfilter, mode = "same")
        dogy = scipy.signal.convolve2d(twod, dyfilter, mode = "same")

        dogdx = scipy.signal.convolve2d(image, dogx, mode="same")
        dogdy = scipy.signal.convolve2d(image, dogy, mode="same")

        doggradientmagnitude = np.sqrt(dogdx ** 2 + dogdy ** 2)

        dogbinarized = doggradientmagnitude > 0.1

        print(np.mean(blurbinarized == dogbinarized))

        plt.imsave("partonepointthreeimages/blur.png", blur, cmap="gray")
        plt.imsave("partonepointthreeimages/blurgradientmagnitude.png", blurgradientmagnitude, cmap="gray")
        plt.imsave("partonepointthreeimages/blurbinarized.png", blurbinarized, cmap="gray")
        plt.imsave("partonepointthreeimages/dogx.png", dogx, cmap="gray")
        plt.imsave("partonepointthreeimages/dogy.png", dogy, cmap="gray")
        plt.imsave("partonepointthreeimages/doggradientmagnitude.png", doggradientmagnitude, cmap="gray")
        plt.imsave("partonepointthreeimages/dogbinarized.png", dogbinarized, cmap="gray")

        
    def parttwopointone():
        image = io.imread("taj.jpg") / 255.0
        nirmit = io.imread("nirmit.jpg")  / 255.0
        landscape = io.imread("twopointoneimage.jpg") / 255.0

        print(image.shape)
        print(nirmit.shape)
        print(landscape.shape)

        oned = cv2.getGaussianKernel(9, 0)
        twod = oned @ oned.T

        i = np.zeros((9, 9))
        i[i.shape[0] // 2, i.shape[1] // 2] = 1

        blurred = np.zeros(image.shape)

        for j in range(3):
            blurred[:, :, j] = scipy.signal.convolve2d(image[:, :, j], twod, mode="same")


        highfrequency = image - blurred  

        unsharpfilter = 2 * i - twod

        sharpened = np.zeros(image.shape)

        for j in range(3):
            sharpened[:, :, j] = scipy.signal.convolve2d(image[:, :, j], unsharpfilter, mode="same")


        plt.imsave("parttwopointoneimages/taj_original.png", image)
        plt.imsave("parttwopointoneimages/taj_blurred.png", blurred)
        plt.imsave("parttwopointoneimages/taj_highfrequency.png", np.clip(highfrequency, 0, 1))
        plt.imsave("parttwopointoneimages/taj_sharpened.png", np.clip(sharpened, 0, 1))

        for amount in [0.25, 0.5, 1, 2, 4, 8]:
            unsharpfilter = (amount + 1) * i - (amount * twod)

            temp = np.zeros(image.shape)

            for j in range(3):
                temp[:, :, j] = scipy.signal.convolve2d(image[:, :, j], unsharpfilter, mode="same")

            

            plt.imsave(f"parttwopointoneimages/taj_sharpened_{amount}.png", np.clip(temp, 0, 1))


        oned = cv2.getGaussianKernel(31, 0)
        twod = oned @ oned.T

        i = np.zeros((31, 31))
        i[i.shape[0] // 2, i.shape[1] // 2] = 1

        nirmitblurred = np.zeros(nirmit.shape)


        for j in range(3):
            nirmitblurred[:, :, j] = scipy.signal.convolve2d(nirmit[:, :, j], twod, mode="same")

        nirmithighfrequency = nirmit - nirmitblurred  



        unsharpfilter = 2 * i - twod

        nirmitsharpened = np.zeros(nirmit.shape)
        
        for j in range(3):
            nirmitsharpened[:, :, j] = scipy.signal.convolve2d(nirmit[:, :, j], unsharpfilter, mode="same")
        

        plt.imsave("parttwopointoneimages/nirmit_original.png", nirmit)
        plt.imsave("parttwopointoneimages/nirmit_blurred.png", nirmitblurred)
        plt.imsave("parttwopointoneimages/nirmit_highfrequency.png", np.clip(nirmithighfrequency, 0, 1))
        plt.imsave("parttwopointoneimages/nirmit_sharpened.png", np.clip(nirmitsharpened, 0, 1))



        landscapeblurred = np.zeros(landscape.shape)

        for j in range(3):
                landscapeblurred[:, :, j] = scipy.signal.convolve2d(landscape[:, :, j], twod, mode="same")
    
        landscapesharpened = np.zeros(landscape.shape)

        for j in range(3):
                landscapesharpened[:, :, j] = scipy.signal.convolve2d(landscapeblurred[:, :, j], unsharpfilter, mode="same")
        
        plt.imsave("parttwopointoneimages/landscape.png", landscape)
        plt.imsave("parttwopointoneimages/landscapeblurred.png", np.clip(landscapeblurred, 0, 1))
        plt.imsave("parttwopointoneimages/landscapesharpened.png", np.clip(landscapesharpened, 0, 1))

        
    def parttwopointtwo():

        # First load images

    

        ## You will provide the code below. Sigma1 and sigma2 are arbitrary 
        ## cutoff values for the high and low frequencies

        def hybrid_image(imhigh, imlow, sigmahigh, sigmalow):

            oned = cv2.getGaussianKernel(31, sigmalow)
            twod = oned @ oned.T

            low = scipy.signal.convolve2d(imlow, twod, mode="same")

            oned = cv2.getGaussianKernel(31, sigmahigh)
            twod = oned @ oned.T

            blurred = scipy.signal.convolve2d(imhigh, twod, mode="same")

            high = imhigh - blurred

            out = low + high

            plt.imsave("parttwopointtwoimages/low.png", low, cmap="gray")
            plt.imsave("parttwopointtwoimages/high.png", high, cmap="gray")

            lowfourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(low))))
            highfourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(high))))

            plt.imsave("parttwopointtwoimages/low_frequency_fourier.png", lowfourier, cmap="gray")
            plt.imsave("parttwopointtwoimages/high_frequency_fourier.png", highfourier, cmap="gray")

            return out

        
        im1 = io.imread("nirmit.jpg", as_gray=True)
        im2 = io.imread("monkey.jpeg", as_gray=True)

        plt.imsave("parttwopointtwoimages/nirmit.png", im1, cmap="gray")
        plt.imsave("parttwopointtwoimages/monkey.png", im2, cmap="gray")

        im1_aligned, im2_aligned = align_images(im1, im2)

        sigma1 = 8
        sigma2 = 4
        hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

        plt.imsave("parttwopointtwoimages/hybrid2.png", hybrid, cmap="gray")


        im1 = io.imread("prof.jpg", as_gray=True)
        im2 = io.imread("goat.jpg", as_gray=True)

        plt.imsave("parttwopointtwoimages/prof.png", im1, cmap="gray")
        plt.imsave("parttwopointtwoimages/goat.png", im2, cmap="gray")

        im1_aligned, im2_aligned = align_images(im1, im2)

        sigma1 = 8
        sigma2 = 4
        hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

        plt.imsave("parttwopointtwoimages/hybrid3.png", hybrid, cmap="gray")

        # high sf
        im1 = io.imread('DerekPicture.jpg', as_gray=True)

        # low sf
        im2 = io.imread('nutmeg.jpg', as_gray=True)


        plt.imsave("parttwopointtwoimages/derek.png", im1, cmap="gray")
        plt.imsave("parttwopointtwoimages/nutmeg.png", im2, cmap="gray")

        # Next align images (this code is provided, but may be improved)
        im1_aligned, im2_aligned = align_images(im1, im2)

        plt.imsave("parttwopointtwoimages/derek_aligned.png", im1_aligned, cmap="gray")
        plt.imsave("parttwopointtwoimages/nutmeg_aligned.png", im2_aligned, cmap="gray")

        derekfourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned))))
        nutmegfourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_aligned))))

        plt.imsave("parttwopointtwoimages/derek_fourier.png", derekfourier, cmap="gray")
        plt.imsave("parttwopointtwoimages/nutmeg_fourier.png", nutmegfourier, cmap="gray")

        sigma1 = 8
        sigma2 = 4
        hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

        plt.imsave("parttwopointtwoimages/hybrid.png", hybrid, cmap="gray")

        hybridfourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid))))
        plt.imsave("parttwopointtwoimages/hybrid_fourier.png", hybridfourier, cmap="gray")



    def parttwopointthree():
        
        apple = io.imread("spline/apple.jpeg")
        orange = io.imread("spline/orange.jpeg")

        oned = cv2.getGaussianKernel(9, 0)
        twod = oned @ oned.T

        applestack = [apple]

        for i in range(4):
            temp = np.zeros(apple.shape)
            for i in range(3):
                temp[:, :, i] = scipy.signal.convolve2d(applestack[-1][:, :, i], twod, mode="same")

            applestack.append(temp)

        applelap = []

        for i in range(4):
            applelap.append(applestack[i] - applestack[i + 1])

        applelap.append(applestack[-1])

        orangestack = [orange]
        
        for i in range(4):
            temp = np.zeros(orange.shape)
            for i in range(3):
                temp[:, :, i] = scipy.signal.convolve2d(orangestack[-1][:, :, i], twod, mode="same")

            orangestack.append(temp)

        orangelap = []

        for i in range(4):
            orangelap.append(orangestack[i] - orangestack[i + 1])

        orangelap.append(orangestack[-1])

    def parttwopointfour():
        apple = io.imread("spline/apple.jpeg")
        orange = io.imread("spline/orange.jpeg")

        oned = cv2.getGaussianKernel(15, 0)
        twod = oned @ oned.T

        depth = 8
        print(apple.shape)
        applestack = [apple]

        for i in range(depth):
            temp = np.zeros(apple.shape)
            for j in range(3):
                temp[:, :, j] = scipy.signal.convolve2d(applestack[-1][:, :, j], twod, mode="same", boundary="symm")

            applestack.append(temp)

        applelap = []

        

        for i in range(depth):
            applelap.append(applestack[i] - applestack[i + 1])

        applelap.append(applestack[-1])

        orangestack = [orange]
        
        for i in range(depth):
            temp = np.zeros(orange.shape)
            for j in range(3):
                temp[:, :, j] = scipy.signal.convolve2d(orangestack[-1][:, :, j], twod, mode="same",  boundary="symm")

            orangestack.append(temp)

        orangelap = []

        for i in range(depth):
            orangelap.append(orangestack[i] - orangestack[i + 1])

        orangelap.append(orangestack[-1])

        mask = np.zeros(apple.shape[:2])

        mask[:, : (apple.shape[1] // 2)] = 1

        maskstack = [mask]

        oned = cv2.getGaussianKernel(41, 0)
        twod = oned @ oned.T

        for i in range(depth):
            
            temp = scipy.signal.convolve2d(maskstack[-1], twod, mode="same",  boundary="symm")

            maskstack.append(temp)



        plt.imshow(maskstack[0], cmap="gray")
        plt.show()

        plt.imshow(maskstack[2], cmap="gray")
        plt.show()

        plt.imshow(maskstack[-1], cmap="gray")
        plt.show()

        benchmark = np.zeros(apple.shape)

        for j in range(3):
            benchmark[:, :, j] = (
                mask * apple[:, :, j]
                +
                (1 - mask) * orange[:, :, j]
            )

        plt.imshow(benchmark.astype(np.uint8))
        plt.show()

        final = []

        for i in range(len(maskstack)):
            temp = np.zeros(apple.shape)
            for j in range(3):
                temp[:, :, j] = (maskstack[i] * applelap[i][:, :, j]) + ((1 - maskstack[i]) * orangelap[i][:, :, j])
            final.append(temp)


        out = np.zeros(apple.shape)

        for i in range(len(final)):
            out += final[i]

        
        plt.imshow(np.clip(out, 0, 255).astype(np.uint8))
        plt.show()

        for level in [0, 2, 4]:

            applepart = np.zeros(apple.shape)
            orangepart = np.zeros(orange.shape)
            combinedpart = np.zeros(apple.shape)

            for j in range(3):
                applepart[:, :, j] = maskstack[level] * applelap[level][:, :, j]
                orangepart[:, :, j] = (1 - maskstack[level]) * orangelap[level][:, :, j]
                combinedpart[:, :, j] = applepart[:, :, j] + orangepart[:, :, j]

            apple_display = applepart - np.min(applepart)
            apple_display = apple_display / np.max(apple_display)

            orange_display = orangepart - np.min(orangepart)
            orange_display = orange_display / np.max(orange_display)

            combined_display = combinedpart - np.min(combinedpart)
            combined_display = combined_display / np.max(combined_display)

            plt.imsave(f"parttwopointfourimages/apple{level}.png", apple_display)
            plt.imsave(f"parttwopointfourimages/orange{level}.png", orange_display)
            plt.imsave(f"parttwopointfourimages/combined{level}.png", combined_display)

        appleout = np.zeros(apple.shape)
        orangeout = np.zeros(orange.shape)

        for i in range(len(maskstack)):
            for j in range(3):
                appleout[:, :, j] += maskstack[i] * applelap[i][:, :, j]
                orangeout[:, :, j] += (1 - maskstack[i]) * orangelap[i][:, :, j]

        
        plt.imsave("parttwopointfourimages/appleout.png", np.clip(appleout, 0, 255).astype(np.uint8))
        plt.imsave("parttwopointfourimages/orangeout.png", np.clip(orangeout, 0, 255).astype(np.uint8))
        plt.imsave("parttwopointfourimages/out.png", np.clip(out, 0, 255).astype(np.uint8))

        sunsetone = cv2.resize(io.imread("sunsetone.jpeg"), (300, 300))
        sunsettwo = cv2.resize(io.imread("sunsettwo.jpeg"), (300, 300))

        sunsetonestack = [sunsetone]

        for i in range(depth):
            temp = np.zeros(sunsetone.shape)
            for j in range(3):
                temp[:, :, j] = scipy.signal.convolve2d(sunsetonestack[-1][:, :, j], twod, mode="same", boundary="symm")

            sunsetonestack.append(temp)

        sunsetonelap = []

        for i in range(depth):
            sunsetonelap.append(sunsetonestack[i] - sunsetonestack[i + 1])

        sunsetonelap.append(sunsetonestack[-1])

        sunsettwostack = [sunsettwo]
        
        for i in range(depth):
            temp = np.zeros(sunsettwo.shape)
            for j in range(3):
                temp[:, :, j] = scipy.signal.convolve2d(sunsettwostack[-1][:, :, j], twod, mode="same", boundary="symm")

            sunsettwostack.append(temp)

        sunsettwolap = []

        for i in range(depth):
            sunsettwolap.append(sunsettwostack[i] - sunsettwostack[i + 1])

        sunsettwolap.append(sunsettwostack[-1])

        mask = np.zeros(sunsetone.shape[:2])

        mask[:, : (sunsetone.shape[1] // 2)] = 1

        maskstack = [mask]

        oned = cv2.getGaussianKernel(31, 0)
        twod = oned @ oned.T

        for i in range(depth):
            
            temp = scipy.signal.convolve2d(maskstack[-1], twod, mode="same", boundary="symm")

            maskstack.append(temp)

        final = []

        for i in range(len(maskstack)):
            temp = np.zeros(sunsetone.shape)
            for j in range(3):
                temp[:, :, j] = (maskstack[i] * sunsetonelap[i][:, :, j]) + ((1 - maskstack[i]) * sunsettwolap[i][:, :, j])
            final.append(temp)

        out = np.zeros(sunsetone.shape)

        for i in range(len(final)):
            out += final[i]

        plt.imsave("parttwopointfourimages/sunsetone.png", sunsetone)
        plt.imsave("parttwopointfourimages/sunsettwo.png", sunsettwo)
        plt.imsave("parttwopointfourimages/sunsetout.png", np.clip(out, 0, 255).astype(np.uint8))
        

        for level in [0, 2, 4]:

            sunsetonepart = np.zeros(sunsetone.shape)
            sunsettwopart = np.zeros(sunsettwo.shape)
            combinedpart = np.zeros(sunsetone.shape)

            for j in range(3):
                sunsetonepart[:, :, j] = maskstack[level] * sunsetonelap[level][:, :, j]
                sunsettwopart[:, :, j] = (1 - maskstack[level]) * sunsettwolap[level][:, :, j]
                combinedpart[:, :, j] = sunsetonepart[:, :, j] + sunsettwopart[:, :, j]

            sunsetone_display = sunsetonepart - np.min(sunsetonepart)
            sunsetone_display = sunsetone_display / np.max(sunsetone_display)

            sunsettwo_display = sunsettwopart - np.min(sunsettwopart)
            sunsettwo_display = sunsettwo_display / np.max(sunsettwo_display)

            combined_display = combinedpart - np.min(combinedpart)
            combined_display = combined_display / np.max(combined_display)

            plt.imsave(f"parttwopointfourimages/sunsetone{level}.png", sunsetone_display)
            plt.imsave(f"parttwopointfourimages/sunsettwo{level}.png", sunsettwo_display)
            plt.imsave(f"parttwopointfourimages/sunsetcombined{level}.png", combined_display)

        sunsetoneout = np.zeros(sunsetone.shape)
        sunsettwoout = np.zeros(sunsettwo.shape)

        for i in range(len(maskstack)):
            for j in range(3):
                sunsetoneout[:, :, j] += maskstack[i] * sunsetonelap[i][:, :, j]
                sunsettwoout[:, :, j] += (1 - maskstack[i]) * sunsettwolap[i][:, :, j]

        
        plt.imsave("parttwopointfourimages/sunsetoneout.png", np.clip(sunsetoneout, 0, 255).astype(np.uint8))
        plt.imsave("parttwopointfourimages/sunsettwoout.png", np.clip(sunsettwoout, 0, 255).astype(np.uint8))
        plt.imsave("parttwopointfourimages/sunsetout.png", np.clip(out, 0, 255).astype(np.uint8))

        
        moon = io.imread("moon.png")[:, :, :3]
        background = io.imread("background.jpeg")

        moon = cv2.resize(moon, (300, 300))
        background = cv2.resize(background, (300, 300))

        moonstack = [moon]

        oned = cv2.getGaussianKernel(15, 0)
        twod = oned @ oned.T

        for i in range(depth):
            temp = np.zeros(moon.shape)
            for j in range(3):
                temp[:, :, j] = scipy.signal.convolve2d(moonstack[-1][:, :, j], twod, mode="same", boundary="symm")

            moonstack.append(temp)

        moonlap = []

        for i in range(depth):
            moonlap.append(moonstack[i] - moonstack[i + 1])

        moonlap.append(moonstack[-1])

        backgroundstack = [background]

        for i in range(depth):
            temp = np.zeros(background.shape)
            for j in range(3):
                temp[:, :, j] = scipy.signal.convolve2d(backgroundstack[-1][:, :, j], twod, mode="same", boundary="symm")

            backgroundstack.append(temp)

        backgroundlap = []

        for i in range(depth):
            backgroundlap.append(backgroundstack[i] - backgroundstack[i + 1])

        backgroundlap.append(backgroundstack[-1])

        mask = np.zeros(moon.shape[:2])

        cv2.circle(mask, (150, 150), 60, 1, -1)

        maskstack = [mask]

        oned = cv2.getGaussianKernel(31, 0)
        twod = oned @ oned.T

        for i in range(depth):

            temp = scipy.signal.convolve2d(maskstack[-1], twod, mode="same", boundary="symm")

            maskstack.append(temp)

        final = []

        for i in range(len(maskstack)):
            temp = np.zeros(moon.shape)
            for j in range(3):
                temp[:, :, j] = (maskstack[i] * moonlap[i][:, :, j]) + ((1 - maskstack[i]) * backgroundlap[i][:, :, j])

            final.append(temp)

        out = np.zeros(moon.shape)

        for i in range(len(final)):
            out += final[i]

        plt.imsave("parttwopointfourimages/moon.png", moon)
        plt.imsave("parttwopointfourimages/background.png", background)
        plt.imsave("parttwopointfourimages/moonmask.png", mask, cmap="gray")
        plt.imsave("parttwopointfourimages/moonout.png", np.clip(out, 0, 255).astype(np.uint8))

    parttwopointfour()